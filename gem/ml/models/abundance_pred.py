from typing import Optional
import torch
from torch import Tensor, nn

from .base import LinearInitializedModule
from .perm_invariant_blocks import ChannelwiseDropout, SumAlongDim, ModelBlock


class SGBEmbedding(LinearInitializedModule):
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
        self.pool = SumAlongDim(dim=-2, keepdim=False)
        self.init_weights(init_rng)

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
        y = y * marker_padding_mask.unsqueeze(-1)  # Zero-out all markers that are "empty markers" (a.k.a. "padding")
        y = self.pool(y)  # shape (*, S, d); "M" gets summed out.
        return y


class SGBAbundancePredictionModel(LinearInitializedModule):
    def __init__(
            self,
            marker_embed_dim: int,
            sgb_embed_latent_dim: int,
            sgb_embed_dim: int,
            n_layers: int,
            model_dim: int,
            num_heads: int,
            key_query_dim: int,
            latent_collection_dim: int,
            combination_latent_dim: int,
            dropout_rate: float = 0.15,
            set_attention_dropout_rate: float = 0.05,
            stochastic_depth_rate: float = 0.1,
            weight_decay_compatible: bool = True,
            init_rng: Optional[torch.Generator] = None
    ):
        """
        :param marker_embed_dim: E
        :param sgb_embed_dim: The target dimension to embed each (M x E) tensor into. Should be sufficiently large to be expressive at the initial embedding step.
        :param n_layers: The number of set-attention-like layers to include.
        :param model_dim: The target dimension of each intermediate stage representation between layers.
        :param num_heads: The number of heads to use for the multi-head attention layers.
        :param key_query_dim: The latent dimension of the attention key and query.
        :param latent_collection_dim: The latent dimension of the set-pool output.
        :param combination_latent_dim: The latent dimension of the perceptron implementing the set-pool & pairwise KQ concatenation.
        :param dropout_rate: Dropout rate applied throughout the model (default: 0.15)
        :param set_attention_dropout_rate: Dropout rate applied throughout the model (default: 0.05)
        :param stochastic_depth_rate: Probability of dropping entire layers during training (default: 0.1)
        :param weight_decay_compatible: Whether to use smaller weight initialization for better weight decay (default: True)
        """
        super().__init__()
        self.model_dim = model_dim

        self.sgb_embedding = SGBEmbedding(marker_embed_dim, sgb_embed_latent_dim, sgb_embed_dim, dropout_rate, init_rng)
        self.initial_linear = nn.Linear(sgb_embed_dim, model_dim)
        self.input_dropout = ChannelwiseDropout(dropout_rate * 0.5)  # Lighter dropout at input

        # Create blocks with increasing stochastic depth rate (more likely to drop later layers)
        block_list = []
        for i in range(n_layers):
            stochastic_depth_rate = stochastic_depth_rate * (i / max(1, n_layers - 1))
            if stochastic_depth_rate == 0.0:
                residual_scale = 1.0
            else:
                residual_scale = 1 / stochastic_depth_rate
            block = ModelBlock(
                model_dim, sgb_embed_dim, num_heads, key_query_dim,
                latent_collection_dim, combination_latent_dim,
                dropout_rate=dropout_rate,
                set_attention_dropout_rate=set_attention_dropout_rate,
                stochastic_depth_rate=stochastic_depth_rate,
                residual_scale=residual_scale,
                init_rng=init_rng
            )
            block_list.append(block)
        self.blocks = nn.ModuleList(block_list)

        self.final_layer_norm = nn.LayerNorm(model_dim)
        self.final_dropout = ChannelwiseDropout(dropout_rate)

        # Final prediction head with additional regularization
        self.final_f = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),  # Reduce dimension for regularization
            nn.GELU(),
            ChannelwiseDropout(dropout_rate),
            nn.Linear(model_dim // 2, 1),
            nn.Flatten(start_dim=-2, end_dim=-1),
        )
        self.init_weights(init_rng, weight_decay_compatible)

    def forward(self, g: Tensor, marker_padding_mask: Tensor, genome_padding_mask: Tensor) -> Tensor:
        """
        :param g: Tensor of shape (n_batch, S, M, E).
        :param marker_padding_mask: Boolean tensor of shape (n_batch, S, M).
        The (i,j,k) entry should be "True" if batch i, SGB j, marker k should be included in computation, "False" otherwise.
        """
        # Start by converting from a larger space into a smaller one.
        sgb = self.sgb_embedding(g, marker_padding_mask)

        # Apply the rest of the layers, including an initial linear projection.
        x = self.initial_linear(sgb)
        x = self.input_dropout(x)
        assert len(x.shape) == 3  # TODO: remove assertion after debugging.
        assert x.shape[0] == g.shape[0]  # TODO: remove assertion after debugging.
        assert x.shape[1] == g.shape[1]  # TODO: remove assertion after debugging.
        assert x.shape[2] == self.model_dim  # TODO: remove assertion after debugging.
        for block in self.blocks:
            x = block(x, sgb, genome_padding_mask)  # shape (B, S, model_dim)

        assert len(x.shape) == 3  # TODO: remove assertion after debugging.
        assert x.shape[0] == g.shape[0]  # TODO: remove assertion after debugging.
        assert x.shape[1] == g.shape[1]  # TODO: remove assertion after debugging.
        assert x.shape[2] == self.model_dim  # TODO: remove assertion after debugging.

        x = self.final_layer_norm(x)  # Final normalization
        x = self.final_dropout(x)
        logits = self.final_f(x)  # shape (B, G) -- operates per-genome
        assert len(logits.shape) == 2  # TODO: remove assertion after debugging.
        assert logits.shape[0] == g.shape[0]  # TODO: remove assertion after debugging.
        assert logits.shape[1] == g.shape[1]  # TODO: remove assertion after debugging.
        assert logits.shape == genome_padding_mask.shape  # TODO: remove assertion after debugging.
        logits = logits.masked_fill(~genome_padding_mask, float("-inf"))
        return logits

    def get_num_parameters(self):
        """Helper method to check model size"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        for block in self.blocks:
            block = torch.utils.checkpoint.checkpoint_wrapper(block)
        return self