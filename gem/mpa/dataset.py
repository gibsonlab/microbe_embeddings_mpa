from typing import *

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .abundance_profile import MetaphlanProfileExtractor, MetaphlanProfile
from .embeddings import MetaphlanMarkerEmbedding
from gem.util.timer import timer


class MetaphlanDataset(Dataset):
    """
    PyTorch Dataset for microbiome data with SGB embeddings and abundance values.
    """

    def __init__(
            self,
            dataset_df: pd.DataFrame,
            marker_embedding: MetaphlanMarkerEmbedding
    ):
        """
        Initialize the microbiome dataset.

        :param dataset_df: A dataframe containing the profile(s) to be included. Index should be
        sample IDs, and each column should be a taxonomic ID.
        :param marker_embedding:
        """
        super().__init__()
        self.df = dataset_df
        self.samples = list(MetaphlanProfileExtractor(dataset_df).samples())

        self.sample_indices = {sample.sample_id: idx for idx, sample in enumerate(self.samples)}
        self.marker_embedding = marker_embedding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sample = self.samples[idx]
        with timer(f"load_sample_embeddings {sample.sample_id}", enabled=False):
            _, features, marker_padding_mask, sgb_padding_mask, targets = self.load_sample_embeddings(sample)
        return features, marker_padding_mask, sgb_padding_mask, targets

    def load_sample_embeddings(self, sample: MetaphlanProfile) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
        """
        Get a single sample, converted into tensors by index.
        Note that each sample's SGB ordering is pre-determined by the input dataframe, so there is no need to return it here.

        :param sample: The sample instance to use.
        :return: Tuple of (features, marker_padding, sgb_padding, targets)
            - sgb_ids: List of the SGB ids that index the tensor.
            - features: Float Tensor of shape (num_sgbs, max_num_markers, embedding_dim) with SGB and their markers' embeddings
            - marker_padding: Boolean-valued Tensor of shape (num_sgbs, max_num_markers) indicating which SGB markers are padding placeholders (False) or actual values (True).
            - sgb_padding: Boolean-valued Tensor of shape (num_sgbs) indicating which SGBs are padding placeholders.
            - targets: Tensor of shape (num_sgbs,) with abundance values
        """
        num_sgbs = len(sample.sgb_ids)
        max_num_markers = max(self.marker_embedding.num_markers(sgb_id) for sgb_id in sample.sgb_ids)
        with timer(f"sample initialization: {sample.sample_id}", enabled=False):
            # Preallocate tensors directly
            features = torch.zeros((num_sgbs, max_num_markers, self.marker_embedding.embedding_dim),
                                   dtype=self.marker_embedding.dtype)
            marker_padding_mask = torch.zeros((num_sgbs, max_num_markers), dtype=torch.bool)
            sgb_padding_mask = torch.zeros(num_sgbs, dtype=torch.bool)

        sgb_ids = []
        targets = torch.zeros(num_sgbs, dtype=features.dtype)
        with timer(f"sample embedding loop over SGB: {sample.sample_id}", enabled=False):
            for sgb_id, sgb_abund in zip(sample.sgb_ids, sample.abundances):
                try:
                    embedding, mask = self.marker_embedding.convert_sgb(sgb_id, max_num_markers)
                except ValueError:
                    print(f"Error while trying to load {sgb_id} for sample {sample.sample_id}")
                    raise
                except KeyError:
                    continue
                else:
                    _i = len(sgb_ids)
                    features[_i] = embedding
                    marker_padding_mask[_i] = mask
                    sgb_padding_mask[_i] = True
                    targets[_i] = sgb_abund
                    sgb_ids.append(sgb_id)

        targets = targets / targets.sum()
        return sgb_ids, features, marker_padding_mask, sgb_padding_mask, targets