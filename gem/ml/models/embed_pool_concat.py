"""
Uses a "shallow" neural network which embeds each marker, (optionally) pools the marker features, and concatenates them.
"""
from typing import Optional

import torch
from torch import Tensor, nn

from .base import LinearInitializedModule
from .perm_invariant_blocks import SumAlongDim


class SGBEmbedPoolConcatPredictionModel(LinearInitializedModule):
    """
    A simple model which applies a MLP to each marker, and pools the markers to form features per species.
    Optionally, applies a second MLP to each species, and pools the species to form sample-wide context feature for the sample.
    If this optional feature is computed, it is concatneated
    """
    def __init__(
            self,
            marker_embed_dim: int,
            sgb_model_dim: int,
            hidden_dim: int,
            use_sgb_pooling: bool,
            sgb_pool_dim: Optional[int] = 0,
            weight_decay_compatible: bool = True,
            init_rng: Optional[torch.Generator] = None,
    ):
        """
        :param weight_decay_compatible:
        :param init_rng:
        """
        super().__init__()
        # self.marker_transform_layer = MarkerEmbedTransform(marker_embed_dim, sgb_model_dim, weight_decay_compatible, init_rng)
        self.marker_transform_layer = nn.Sequential(
            nn.Linear(marker_embed_dim, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sgb_model_dim),
            nn.LayerNorm(normalized_shape=sgb_model_dim),
            nn.GELU(),
        )
        self.marker_pool_layer = SumAlongDim(dim=-2, keepdim=False)

        self.use_sgb_pooling = use_sgb_pooling
        if use_sgb_pooling:
            assert sgb_pool_dim > 0, "If pooling is turned on, sgb_pool_dim must be specified and greater than 0."
            self.species_transform_layer = nn.Sequential(
                nn.Linear(sgb_model_dim, hidden_dim),
                nn.LayerNorm(normalized_shape=hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, sgb_pool_dim),
                nn.LayerNorm(normalized_shape=sgb_pool_dim),
                nn.GELU(),
            )
            self.species_pool_layer = SumAlongDim(dim=-2, keepdim=False)

        # define final layer.
        if use_sgb_pooling:
            prediction_input_dim = sgb_model_dim + sgb_pool_dim  # concatentaed dim
        else:
            prediction_input_dim = sgb_model_dim

        self.prediction_layer = nn.Sequential(
            nn.Linear(prediction_input_dim, hidden_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Flatten(start_dim=-2, end_dim=-1),
        )

        self.init_weights(init_rng, weight_decay_compatible)

    def forward(self, g: Tensor, marker_padding_mask: Tensor, sgb_padding_mask: Tensor) -> Tensor:
        """
        :param g: Tensor of shape (n_batch, S, M, E).
        :param marker_padding_mask: Boolean tensor of shape (n_batch, S, M). The (i,j,k) entry should be "True" if batch i, SGB j, marker k should be included in computation, "False" otherwise.
        :param sgb_padding_mask: Boolean tensor of shape (n_batch, S). the (i,j) entry should be "True" if batch_i, SGB j should be included in computation.
        """
        x = self.marker_transform_layer(g)                                     # shape (*, S, M, sgb_model_dim)
        x = self.marker_pool_layer(x * marker_padding_mask.unsqueeze(-1))      # shape (*, S, sgb_model_dim)

        if self.use_sgb_pooling:
            y = self.species_transform_layer(x)                                    # shape (*, S, sgb_pool_dim)
            y = self.species_pool_layer(y * sgb_padding_mask.unsqueeze(-1))        # shape (*, sgb_pool_dim)
            y = y.expand(*x.shape[:-1], -1)                                        # shape (*, S, sgb_pool_dim), broadcasted along dim=-2

            xy = torch.concatenate([x, y], dim=-1)                          # shape (*, S, sgb_pool_dim + sgb_model_dim)
            logits = self.prediction_layer(xy)                                     # shape (*, S)
        else:
            logits = self.prediction_layer(x)                                      # shape (*, S)

        logits = logits.masked_fill(~sgb_padding_mask, float("-inf"))
        return logits
