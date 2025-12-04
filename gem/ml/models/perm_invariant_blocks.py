""" Classes copied from other notebook (american_gut) """
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from .base import LinearInitializedModule


class ChannelwiseDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Dropout that applies the same dropout mask across all elements of the first (n-1) dimensions
        for each channel in the last dimension. Preserves permutation invariance.

        :param p: probability of an element to be zeroed (dropout rate).
        """
        super(ChannelwiseDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # only perform dropouts in "training" mode, not "eval" mode.
        if not self.training or self.p == 0.0:
            return x

        shape = x.shape

        # Create dropout mask of shape (k,)
        # (1-p) is probability to keep, so apply Bernoulli with prob (1-p)
        device = x.device
        mask = torch.bernoulli(torch.ones(shape[-1], device=device) * (1 - self.p))
        mask = mask / (
                    1 - self.p)  # Scale mask to preserve expectation, see "inverted dropout", also performed in torch's built-in dropout.
        mask = mask.view(*(1 for _ in shape[:-1]), shape[-1])  # Reshape mask to broadcast over (n, m, l)
        return x * mask


class SumAlongDim(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x) -> Tensor:
        return x.sum(dim=self.dim, keepdim=self.keepdim)


class MatrixConcatBroadcastFFN(LinearInitializedModule):
    def __init__(self, embed_dim: int, hidden_dim: int, init_rng: Optional[torch.Generator] = None):
        """
        A model which takes as input two tensors:
        - `A` of shape (*, G, G)
        - `B` of shape (*, embed_dim).
        This module implemented the broadcasted concatenation of B onto A, producing a tensor of shape (*, G, G, 1+embed_dim); B[*,:] is broadcasted into every entry A[*,i,j].
        Then, this is passed into a feedforward network taking a (*, G, G, 1+embed_dim) as input, and outputting a (*, G, G, 1) tensor.
        The last dim is squeezed, producing an output of shape (*, G, G).
        """
        super().__init__()
        self.linear_A = nn.Linear(1, hidden_dim)
        self.linear_B = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.final_linear = nn.Linear(hidden_dim, 1)
        self.init_weights(init_rng)

    def forward(self, A: Tensor, B: Tensor) -> Tensor:
        # A: (*, G, G)
        # B: (*, embed_dim)
        A_proj = self.linear_A(A.unsqueeze(-1))  # (*, G, G, hidden_dim)
        B_proj = self.linear_B(B).unsqueeze(-2).unsqueeze(-2)  # (*, 1, 1, hidden_dim)
        y = self.activation(A_proj + B_proj)  # (*, G, G, hidden_dim)
        y = self.final_linear(y).squeeze(-1)  # (*, G, G)
        return y


class MultiHeadSetPool(LinearInitializedModule):
    """
    Lifted directly from other notebook (american_gut).

    Implements a "set-pool" operation with multiple heads: head_k = pool(phi(x1), phi(x2), ..., phi(xk)),
    where xi is the i-th SGB's embedded feature vector. Each xi is assumed to have dimension `genome_feature_dim`.

    More concretely, this layer implements the (latent) representation of each sample as a set of SGBs,
    where the output is the set-pool operation of latent genomic features.

    Guarantees: Output is invariant (up to numerical precision) to permutations of SGBs, and invariant to
    permutations of markers of each SGB.
    """

    def __init__(
            self,
            genome_feature_dim: int,
            out_dim_per_head: int,
            num_heads: int,
            dropout_rate: float = 0.1,
            init_rng: Optional[torch.Generator] = None
    ):
        super().__init__()
        self.linear = nn.Linear(genome_feature_dim, num_heads * out_dim_per_head)
        self.activation1 = nn.GELU()
        self.symmetric_dropout = ChannelwiseDropout(dropout_rate)
        self.unflatten = nn.Unflatten(
            dim=-1, unflattened_size=[num_heads, out_dim_per_head]
        )
        self.pool = SumAlongDim(dim=-3, keepdim=False)
        self.init_weights(init_rng)

    def forward(self, g: Tensor, genome_padding_mask: Tensor):
        """
        :param g: A float tensor of shape (*, G, genome_dim).
        :param genome_padding_mask: a boolean tensor of shape (*, G). Value [i,j] is "true" if the batch i, genome j is a non-empty organism.
        """
        assert len(genome_padding_mask.shape) == 2  # TODO: remove assertion after debugging.
        assert g.shape[0] == genome_padding_mask.shape[0]  # TODO: remove assertion after debugging.
        assert g.shape[1] == genome_padding_mask.shape[1]  # TODO: remove assertion after debugging.

        # Let D = out_dim_per_head, H=# heads.
        y = self.linear(g)  # shape (n_batch=B, n_genomes=G, H*D); per-genome operation.
        y = self.activation1(y)  # shape (B, G, H*D); element-by-element symmetric operation
        y = self.symmetric_dropout(y)  # per-genome operation (using special dropout class)
        y = y * genome_padding_mask.unsqueeze(-1)  # Zero-out all genomes that are "empty genomes" (a.k.a. "padding")
        y = self.unflatten(y)  # shape (B, G, H, D); per-genome operation.
        y = self.pool(y)  # shape (B, H, D); "G" gets summed out.
        return y


class MultiHeadUnit(LinearInitializedModule):
    """ Lifted directly from other notebook (american_gut) """

    def __init__(
            self,
            model_dim: int,
            genome_dim: int,
            num_heads: int,
            key_query_dim: int,
            latent_collection_dim: int,
            combination_latent_dim: int,
            set_attention_dropout_rate: float = 0.05,
            output_dropout_rate: float = 0.1,
            init_rng: Optional[torch.Generator] = None
    ):
        super().__init__()
        self.genome_dim = genome_dim
        self.model_dim = model_dim

        self.collection_pool = MultiHeadSetPool(
            genome_feature_dim=genome_dim,
            num_heads=num_heads,
            out_dim_per_head=latent_collection_dim,
            dropout_rate=set_attention_dropout_rate,
            init_rng=init_rng
        )

        self.key_proj = nn.Sequential(
            nn.Linear(genome_dim, num_heads * key_query_dim),
            nn.Unflatten(dim=-1, unflattened_size=[num_heads, key_query_dim])
        )
        self.query_proj = nn.Sequential(
            nn.Linear(model_dim, num_heads * key_query_dim),
            nn.Unflatten(dim=-1, unflattened_size=[num_heads, key_query_dim])
        )
        self.norm_const = 1 / float(np.sqrt(key_query_dim))
        self.kq_activation = nn.GELU()

        self.combination = MatrixConcatBroadcastFFN(
            embed_dim=latent_collection_dim,
            hidden_dim=combination_latent_dim,
            init_rng=init_rng
        )
        self.head_linear = nn.Linear(num_heads, model_dim)
        self.output_dropout = ChannelwiseDropout(output_dropout_rate)
        self.init_weights(init_rng)

    def forward(self, x: Tensor, g: Tensor, genome_padding_mask: Tensor):
        """
        Generally, the inputs are expected to have the following shapes:
            - x: (n_batch, n_genomes, model_dim)
            - g: (n_batch, n_genomes, genome_dim)
            - genome_padding_mask: (n_batch, n_genomes) boolean tensor
        """
        assert len(g.shape) == 3  # TODO: remove assertion after debugging.
        assert len(x.shape) == 3  # TODO: remove assertion after debugging.
        assert x.shape[0] == g.shape[
            0], f"# of batches do not match. x: {x.shape[0]}, g: {g.shape[0]}"  # TODO: remove assertion after debugging.
        assert x.shape[1] == g.shape[
            1], f"# of genomes do not match. x: {x.shape[1]}, g: {g.shape[1]}"  # TODO: remove assertion after debugging.

        c = self.collection_pool(g,
                                 genome_padding_mask)  # Combines genomes & permutation-invariant; ZEROES-out "empty genomes" using masks. Shape (B, latent_collection_dim, H)
        k = self.key_proj(
            g)  # Operaters per-genome. Shape (B, G, H, key_query_dim); permutation-respecting along n_genomes dimension; e.g. key_proj(perm(g)) = perm(key_proj(g))
        q = self.query_proj(
            x)  # Operaters per-genome. Shape (B, G, H, key_query_dim); permutation-respecting along n_genomes dimension; e.g. query_proj(perm(g)) = perm(query_proj(g))

        # kq: permutation-respecting along both n_genomes dimensions.
        kq = self.norm_const * torch.matmul(k.permute(0, 2, 1, 3), q.permute(0, 2, 3,
                                                                             1))  # shape (B, H, G, G)  -> Computes the key-query combination for each pair of genomes, per head. rows are indexed by key genome and columns indexed by query "x".
        kq = self.kq_activation(kq)  # shape (B, H, G, G)

        y = self.combination(kq, c)  # shape (B, H, G, G)

        # apply masking.
        # Expand m for broadcasting to target positions i and j.
        mask_i = genome_padding_mask.unsqueeze(2)  # shape (B, G, 1)
        mask_j = genome_padding_mask.unsqueeze(1)  # shape (B, 1, G)
        mask_ij = mask_i & mask_j  # shape (B, G, G)
        mask_ij = mask_ij.unsqueeze(1)  # shape (B, 1, G, G)

        y = y * mask_ij  # shape (B, H, G, G), but entries (batch, head, i, j) have been zeroed out if mask[batch, i] == 0 or mask[batch, j] == 0.
        y = y.sum(dim=-1)  # shape (B, H, G)
        y = y.permute(0, 2, 1)  # shape (B, G, H)
        y = self.head_linear(y)  # shape (B, G, model_dim)
        y = self.output_dropout(y)  # per-genome operation, using special dropout class.
        return y


class ModelBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            genome_dim: int,
            num_heads: int,
            key_query_dim: int,
            latent_collection_dim: int,
            combination_latent_dim: int,
            residual_scale: float = 0.1,
            set_attention_dropout_rate: float = 0.05,
            dropout_rate: float = 0.1,
            stochastic_depth_rate: float = 0.0,
            init_rng: Optional[torch.Generator] = None
    ):
        super().__init__()
        self.unit = MultiHeadUnit(
            model_dim, genome_dim, num_heads, key_query_dim,
            latent_collection_dim, combination_latent_dim,
            set_attention_dropout_rate=set_attention_dropout_rate,
            output_dropout_rate=dropout_rate,
            init_rng=init_rng
        )
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            ChannelwiseDropout(dropout_rate),
            nn.Linear(model_dim, model_dim),
            ChannelwiseDropout(dropout_rate),
        )

        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.residual_scale = residual_scale
        self.stochastic_depth_rate = stochastic_depth_rate
        self._init_weights(init_rng)

    def _init_weights(self, init_rng: Optional[torch.Generator] = None):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5, generator=init_rng)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _stochastic_depth(self, x, residual):
        """Apply stochastic depth (layer dropout) during training"""
        if not self.training or self.stochastic_depth_rate == 0.0:
            return residual + self.residual_scale * x

        # Random survival probability
        if torch.rand(1).item() > self.stochastic_depth_rate:
            return residual + self.residual_scale * x
        else:
            return residual

    def forward(self, x: Tensor, g: Tensor, genome_padding_mask: Tensor) -> Tensor:
        # Note: since most operations here are per-genome, padding_mask need not be applied explicitly. We pass that responsibility into the succeeding layers that come after this block.
        # The only exception is the MultiHeadUnit, so we also pass the responsibility of masking onto that layer.
        residual = x
        x = self.layer_norm1(x)  # operates per-genome
        x = self.unit(x, g, genome_padding_mask)  # permutation-invariant.
        x = self.layer_norm2(x)  # operates per-genome
        x = self.feedforward(x)  # operates per-genome
        x = self._stochastic_depth(x, residual)  # operates per-genome (residual connection for gradient stability)
        return x