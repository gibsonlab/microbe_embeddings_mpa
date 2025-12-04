from typing import *

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .abundance_profile import MetaphlanProfileExtractor, MetaphlanProfile
from .embeddings import MetaphlanMarkerEmbedding
from gem.util.timer import timer


def load_dataset_dataframes(profile_tsv: Path, metadata_tsv: Path):
    profiles = pd.read_csv(profile_tsv, sep="\t")
    profiles_indexed = profiles.set_index("clade_name").transpose()
    metadata = pd.read_csv(metadata_tsv, sep="\t")
    return profiles_indexed, metadata


class MetaphlanDataset(Dataset):
    """
    PyTorch Dataset for microbiome data with SGB embeddings and abundance values.
    """

    def __init__(
            self,
            dataset_df: pd.DataFrame,
            marker_embedding: MetaphlanMarkerEmbedding,
            max_num_sgbs: Optional[int] = None
    ):
        """
        Initialize the microbiome dataset.

        :max_num_sgbs: Specify the number of SGB slots to produce in each feature tensor.
        Note: if max_num_sgbs is smaller than the actual maximum in the samples, then those
        samples exceeding this many SGBs will be removed.
        :param dataset_df: A dataframe containing the profile(s) to be included. Index should be
        sample IDs, and each column should be a taxonomic ID.
        """
        self.df = dataset_df
        self.samples = list(MetaphlanProfileExtractor(dataset_df).samples())

        if max_num_sgbs is not None:
            print(f"Enforcing num_sgbs <= {max_num_sgbs}...")
            self.samples = [s for s in self.samples if len(s.sgb_ids) <= max_num_sgbs]
            print(f"--> Filtered out {len(self.samples)} samples out of {dataset_df.shape[0]}.")
            self.max_num_sgbs = max_num_sgbs
        else:
            self.max_num_sgbs = max(len(sample.sgb_ids) for sample in self.samples)

        self.sample_indices = {sample.sample_id: idx for idx, sample in enumerate(self.samples)}

        self.max_num_markers = marker_embedding.max_num_markers
        print("Dataset statistics:")
        print(f"\tMax. # SGBs = {self.max_num_sgbs}")
        print(f"\tMax. # SGB markers = {self.max_num_markers}")
        self.marker_embedding = marker_embedding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sample = self.samples[idx]
        with timer(f"load_sample_embeddings {sample.sample_id}"):
            _, features, marker_padding_mask, sgb_padding_mask, targets = self.load_sample_embeddings(sample)
        return features, marker_padding_mask, sgb_padding_mask, targets

    def load_sample_embeddings(self, sample: MetaphlanProfile) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
        """
        Get a single sample, converted into tensors by index.
        Note that each sample's SGB ordering is pre-determined by the input dataframe, so there is no need to return it here.

        :param sample: The sample instance to use.
        :return: Tuple of (features, padding, targets)
            - features: Float Tensor of shape (max_num_sgbs, max_num_markers, embedding_dim) with SGB and their markers' embeddings
            - marker_padding_mask: Boolean-valued Tensor of shape (max_num_sgbs, max_num_markers) indicating which SGBs and/or SGB-markers are padding placeholders (False) or actual values (True).
            - targets: Tensor of shape (max_num_sgbs,) with abundance values
        """

        with timer(f"sample initialization: {sample.sample_id}"):
            # Preallocate tensors directly
            features = torch.zeros((self.max_num_sgbs, self.max_num_markers, self.marker_embedding.embedding_dim),
                                   dtype=self.marker_embedding.dtype)
            marker_padding_mask = torch.zeros((self.max_num_sgbs, self.max_num_markers), dtype=torch.bool)
            sgb_padding_mask = torch.zeros(self.max_num_sgbs, dtype=torch.bool)

        sgb_ids = []
        with timer(f"sample embedding loop over SGB: {sample.sample_id}"):
            for sgb_id in sample.sgb_ids:
                try:
                    embedding, mask = self.marker_embedding.convert_sgb(sgb_id, self.max_num_markers)
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
                    sgb_ids.append(sgb_id)

        with timer(f"sample tensor creation: {sample.sample_id}"):
            targets = torch.zeros(self.max_num_sgbs, dtype=features.dtype)
            sample_abunds = sample.abundances
            sample_abunds = sample_abunds / np.sum(sample_abunds)
            targets[:len(sample.sgb_ids)] = torch.as_tensor(sample_abunds, dtype=features.dtype)

        return sgb_ids, features, marker_padding_mask, sgb_padding_mask, targets