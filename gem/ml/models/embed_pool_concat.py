"""
Uses a "shallow" neural network which embeds each marker, (optionally) pools the marker features, and concatenates them.
"""
from typing import Optional

import torch
from torch import Tensor, nn

from .base import LinearInitializedModule
from .perm_invariant_blocks import SumAlongDim, ChannelwiseDropout


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, add_residual=False):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(normalized_shape=output_dim)
        self.gelu = nn.GELU()
        self.dropout = ChannelwiseDropout(dropout_rate)
        self.add_residual = add_residual

    def forward(self, x):
        identity = x
        x = self.fc(x)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.dropout(x)
        if self.add_residual:
            print(identity.shape)
            print(x.shape)
            return x + identity  # Residual connection
        else:
            return x


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
            dropout_rate: float = 0.0,
            weight_decay_compatible: bool = True,
            init_rng: Optional[torch.Generator] = None,
    ):
        """
        :param weight_decay_compatible:
        :param init_rng:
        """
        super().__init__()
        print(f"Initializing model with dropout_rate = {dropout_rate}")
        # self.marker_transform_layer = MarkerEmbedTransform(marker_embed_dim, sgb_model_dim, weight_decay_compatible, init_rng)
        self.marker_transform_layer = nn.Sequential(
            ResidualBlock(marker_embed_dim, hidden_dim, dropout_rate, add_residual=False),
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate, add_residual=True),
            ResidualBlock(hidden_dim, sgb_model_dim, 0.0, add_residual=False),
        )

        self.use_sgb_pooling = use_sgb_pooling
        if use_sgb_pooling:
            assert sgb_pool_dim > 0, "If pooling is turned on, sgb_pool_dim must be specified and greater than 0."

        self.species_transform_layer = nn.Sequential(
            ResidualBlock(sgb_model_dim, hidden_dim, dropout_rate, add_residual=False),
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate, add_residual=True),
            ResidualBlock(hidden_dim, sgb_pool_dim, 0.0, add_residual=False),
        )

        # define final layer.
        if use_sgb_pooling:
            prediction_input_dim = sgb_model_dim + sgb_pool_dim  # concatentaed dim
        else:
            prediction_input_dim = sgb_model_dim

        self.prediction_layer = nn.Sequential(
            ResidualBlock(prediction_input_dim, hidden_dim, dropout_rate, add_residual=False),
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate, add_residual=True),
            ResidualBlock(hidden_dim, 1, 0.0, add_residual=False),
        )
        self.final_reshape_layer = nn.Flatten(start_dim=-2, end_dim=-1)

        self.init_weights(init_rng, weight_decay_compatible)

    def forward(self, g: Tensor, marker_padding_mask: Tensor, sgb_padding_mask: Tensor) -> Tensor:
        """
        :param g: Tensor of shape (n_batch, S, M, E).
        :param marker_padding_mask: Boolean tensor of shape (n_batch, S, M). The (i,j,k) entry should be "True" if batch i, SGB j, marker k should be included in computation, "False" otherwise.
        :param sgb_padding_mask: Boolean tensor of shape (n_batch, S). the (i,j) entry should be "True" if batch_i, SGB j should be included in computation.
        """
        x = self.marker_transform_layer(g)                                         # shape (*, S, M, sgb_model_dim)
        # ========== mean-pooling (nanmean)
        x = torch.sum(
            x * marker_padding_mask.unsqueeze(-1),
            dim=-2, keepdim=False
        )                                                                          # shape (*, S, sgb_model_dim)
        num_markers = marker_padding_mask.sum(dim=-1, keepdim=True).clamp(min=1)   # shape (*, S, 1)
        x = x / num_markers                                                        # shape (*, S, sgb_model_dim)

        if self.use_sgb_pooling:
            print("here-1")
            y = self.species_transform_layer(x)                                    # shape (*, S, sgb_pool_dim)
            print("here0")
            # ========== mean-pooling (nanmean)
            y = torch.sum(
                y * sgb_padding_mask.unsqueeze(-1),
                dim=-2, keepdim=False
            )                                                                      # shape (*, sgb_pool_dim)
            print("here1")
            num_species = sgb_padding_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # shape (*, 1)
            y = y / num_species                                                    # shape (*, sgb_pool_dim)

            y = y.unsqueeze(-2)                                                    # shape (*, 1, sgb_pool_dim)
            y = y.expand(*x.shape[:-1], y.shape[-1])                                        # shape (*, S, sgb_pool_dim), broadcasted along dim=-2

            print("here2:", y.shape)
            xy = torch.concatenate([x, y], dim=-1)                          # shape (*, S, sgb_pool_dim + sgb_model_dim)
            print("here2:", xy.shape)
            logits = self.prediction_layer(xy)                                    # shape (*, S)
            print("here3:", logits.shape)
        else:
            logits = self.prediction_layer(x)                                      # shape (*, S)

        print("here4:", logits.shape)
        print(sgb_padding_mask.shape)
        logits = self.final_reshape_layer(logits)
        logits = logits.masked_fill(~sgb_padding_mask, float("-inf"))
        print("Here5: {}".format(logits.shape))
        return logits
