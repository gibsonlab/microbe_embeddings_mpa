from typing import *

import torch
from torch import Tensor, nn

from .base import LinearInitializedModule
from .perm_invariant_blocks import ChannelwiseDropout, SumAlongDim


class SGBEmbeddingRenormalized(LinearInitializedModule):
    """
    A class meant to embed each SGB per input sample, where each SGB itself is a set (size=M) of markers,
    each marker is a genomic sequence embedding (dimension=E).

    Note that this is meant to be a conversion for each SGB from a (M * E) space into a smaller-dim
    space of dimension (d) of latent genomic features.

    Guarantees: Each genomic feature (i=1 to d) should be invariant under marker permutation.
    """

    def __init__(
            self,
            input_embed_dim: int,  # E
            latent_dim: int,  # h
            out_dim: int,  # d
            dropout_rate: float = 0.1,
            init_rng: Optional[torch.Generator] = None
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=input_embed_dim, out_features=latent_dim)
        self.activation1 = nn.GELU()
        self.linear2 = nn.Linear(in_features=latent_dim, out_features=out_dim)
        self.activation2 = nn.GELU()
        self.symmetric_dropout = ChannelwiseDropout(dropout_rate)
        self.init_weights(init_rng)
        self.mask_renormalization_eps = 1e-5

    def forward(self, x: Tensor, marker_padding_mask: Tensor) -> Tensor:
        """
        :param x: A float tensor of shape (*, S, M, E).
        :param marker_padding_mask: a boolean mask tensor of shape (*, S, M). Value [*, i, j] is "false" if marker j of SGB i is a padding/empty marker.
        :return: Tensor of shape (*, S, H, d).
        """
        y = self.linear(x)  # shape (*, S, M, h); per-marker operation.
        y = self.activation1(y)  # shape (*, S, M, h); element-by-element symmetric operation
        y = self.linear2(y)  # shape (*, S, M, d); per-marker operation.
        y = self.activation2(y)  # shape (*, S, M, d); element-by-element symmetric operation
        y = self.symmetric_dropout(y)  # shape (*, S, M, d); per-genome operation (using special dropout class)

        # compute the masked average.
        y_masked = y * marker_padding_mask.unsqueeze(-1)           # shape (*, S, M, d), zero-out all markers with mask "False"
        y_sum = y_masked.sum(dim=-2)                               # shape (*, S, d)
        mask_count = marker_padding_mask.sum(dim=-1, keepdim=True) # shape (*, S, 1)
        y = y_sum / (mask_count + self.mask_renormalization_eps)   # shape (*, S, d)
        return y


class L1Normalize(nn.Module):
    def __init__(self, dim=-1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        # x: (*, M, N)
        return x / (x.abs().sum(dim=self.dim, keepdim=True) + self.eps)


class V2Layer(LinearInitializedModule):
    def __init__(
            self,
            sgb_marker_embed_dim: int,   # E, the input genome embedding feature dimension
            sgb_model_dim: int,          # D, the output dimension per SGB
            sgb_proj_dim_per_head: int,  # smaller projection dim per head
            num_heads: int,
            sgb_embed_dropout_rate: float = 0.1,
            weight_decay_compatible: bool = True,
            init_rng: Optional[torch.Generator] = None,
    ):
        super().__init__()

        # Each layer has its own SGB embedding & embedding projection.
        self.sgb_embedding = SGBEmbeddingRenormalized(
            input_embed_dim=sgb_marker_embed_dim,
            latent_dim=sgb_model_dim * 2,
            out_dim=sgb_model_dim,
            dropout_rate=sgb_embed_dropout_rate,
        )
        self.sgb_embedding_projection_multihead = nn.Sequential(
            nn.Linear(sgb_model_dim, sgb_proj_dim_per_head * num_heads, bias=False),
            nn.Unflatten(dim=-1, unflattened_size=[num_heads, sgb_proj_dim_per_head])
        )
        self.y_projection_multihead = nn.Sequential(
            nn.Linear(sgb_model_dim, sgb_proj_dim_per_head * num_heads, bias=False),
            nn.Unflatten(dim=-1, unflattened_size=[num_heads, sgb_proj_dim_per_head])
        )
        self.heads_flatten = nn.Flatten(start_dim=-2, end_dim=-1)

        self.head_concat_feedforward = nn.Sequential(
            nn.Linear(in_features=num_heads * sgb_proj_dim_per_head, out_features=sgb_model_dim),
            nn.GELU()
        )
        self.head_collapse_feedforward = nn.Sequential(
            nn.Linear(in_features=num_heads, out_features=1),
            nn.GELU()
        )
        self.y2_batch_renorm = nn.LayerNorm(normalized_shape=num_heads * sgb_proj_dim_per_head)
        self.sgb_mask_eps = 1e-5

        # self.product_activation = nn.Softmax(dim=-1)
        self.gelu = nn.GELU()
        self.product_activation = nn.Tanh()

        self.l1_renorm = L1Normalize(dim=-1, eps=1e-5)
        self.a_bias = nn.Parameter(torch.zeros(1))
        self.a_scale = nn.Parameter(torch.ones(1))
        self.a_activation = nn.PReLU()
        self.init_weights(init_rng, weight_decay_compatible)

    def forward(
            self,
            g: Tensor,
            Y: Tensor,
            marker_padding_mask: Tensor,
            sgb_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        :param g: A (*, S, M, E) tensor of genome embeddings. Each i-th slice along dim "-2" is the input embedding of the i-th input SGB.
        :param Y: A (*, S, D) tensor of current latent SGB representation. Each i-th slice is the order-k way products ("interactions") between k-1 taxa and the i-th SGB.
        :param sgb_padding_mask: A (*, S) tensor of boolean masks.
        :return:
        """
        sgb_mask_expanded = sgb_padding_mask.unsqueeze(-1).unsqueeze(-1)  # shape (*, S, 1, 1)
        sgb_dot_rescaling = (
                torch.sqrt(sgb_mask_expanded.sum(dim=-3, keepdim=True))
                + self.sgb_mask_eps
        ) # shape (*, 1, 1, 1)

        X = self.sgb_embedding(g, marker_padding_mask)             # shape (*, S, D)
        X = self.sgb_embedding_projection_multihead(X)             # shape (*, S, H, d_head), Linear
        X = X * sgb_mask_expanded                                  # shape (*, S, H, d_head) --> mask each slice (b, s, *)
        X = X.transpose(-2, -3)                                    # shape (*, H, S, d_head)

        Y_proj = self.y_projection_multihead(Y)                    # shape (*, S, H, d_head), Linear
        Y_proj = Y_proj * sgb_mask_expanded                        # shape (*, S, H, d_head) --> mask each slice (b, s, *)
        Y_proj = Y_proj.transpose(-2, -3)                          # shape (*, H, S, d_head)

        # First computation of Y&Z product occurs here.
        # Note: mask-based sqrt-factor rescaling is done here.
        Z = X @ Y_proj.transpose(-2, -1) / sgb_dot_rescaling       # shape (*, H, S, S)
        Z = self.product_activation(Z)                             # shape (*, H, S, S), nonlinear: last dimension sums to 1.0

        Y2 = Z @ X                                                 # shape (*, H, S, d_head), convex polytope in row space of X
        Y2 = self.gelu(Y2)                                         # same shape, nonlinearity.
        Y2 = Y2.transpose(-2, -3)                                  # shape (*, S, H, d_head), reshape
        Y2 = self.heads_flatten(Y2)                                # shape (*, S, H*d_head), reshape
        Y2 = self.y2_batch_renorm(Y2)
        Y2 = self.head_concat_feedforward(Y2)                      # shape (*, S, D), linear with nonlinear activation

        # for each (sgb_i, z_j) pair, collapse all the heads.
        A = torch.diagonal(Z, dim1=-2, dim2=-1)                    # shape (*, H, S)
        A = A.transpose(-1, -2)                                    # shape (*, S, H)
        A = self.head_collapse_feedforward(A)                      # shape (*, S, 1), linear with nonlinear activation
        assert A.shape[-1] == 1, "Expected last dim to be size 1, to remove using squeeze."
        A = A.squeeze(dim=-1)                                      # shape (*, S)
        A = self.a_activation(self.a_bias + A * self.a_scale)      # final rescaling by constant factor, learnable.
        # Note: the final scale (a_activation Prelu kernel scale, and a_scale) need to be learned on a per-layer basis!
        # This is intentional -- deeper layers represent higher-order interactions, and may have smaller contribution.

        return A, Y2


class SGBAbundanceLayeredPredictionModel(LinearInitializedModule):
    def __init__(
            self,
            num_layers: int,
            sgb_model_dim: int,
            sgb_marker_embed_dim: int,
            layer_num_heads: int,
            sgb_proj_dim_per_head: int,
            weight_decay_compatible: bool = True,
            init_rng: Optional[torch.Generator] = None,
    ):
        """

        :param num_layers: The number of layers to use. 1 layer is akin to doing per-
        :param weight_decay_compatible:
        :param init_rng:
        """
        super().__init__()
        self.sgb_model_dim = sgb_model_dim
        layer_list = [
            V2Layer(
                sgb_marker_embed_dim=sgb_marker_embed_dim,
                sgb_model_dim=sgb_model_dim,
                sgb_proj_dim_per_head=sgb_proj_dim_per_head,
                num_heads=layer_num_heads,
                weight_decay_compatible=weight_decay_compatible,
                init_rng=init_rng,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(layer_list)
        self.init_weights(init_rng, weight_decay_compatible)

    def forward(self, g: Tensor, marker_padding_mask: Tensor, sgb_padding_mask: Tensor) -> Tensor:
        """
        :param g: Tensor of shape (n_batch, S, M, E).
        :param marker_padding_mask: Boolean tensor of shape (n_batch, S, M). The (i,j,k) entry should be "True" if batch i, SGB j, marker k should be included in computation, "False" otherwise.
        :param sgb_padding_mask: Boolean tensor of shape (n_batch, S). the (i,j) entry should be "True" if batch_i, SGB j should be included in computation.
        """
        # A: "cumulative" logits
        A = torch.zeros_like(sgb_padding_mask)

        # Y: "k-wise" embedding.
        Y = torch.ones(g.shape[:-2] + (self.sgb_model_dim,), dtype=g.dtype, device=g.device)    # shape (*, S, D)

        for i, layer in enumerate(self.layers):
            A_delta, Y = layer(g, Y, marker_padding_mask, sgb_padding_mask)

            A = A + A_delta  # add the contribution from higher-order terms

        logits = A.masked_fill(~sgb_padding_mask, float("-inf"))
        return logits