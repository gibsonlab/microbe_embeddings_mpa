from typing import *
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .samples import MicrobiomeSample, MicrobiomeProject


class MicrobiomeSampleEmbedding:
    def __init__(
            self,
            embedding_file: Path,
            cache_embeddings: bool = False,
    ):
        self.embedding_file = embedding_file

        with h5py.File(self.embedding_file, "r") as h5_file:
            self.asv_id_subset = set(h5_file.keys())

            first_key = list(h5_file.keys())[0]
            example_embedding = h5_file[first_key][:]
            print("Loading tensor embeddings from {}. Embedding shape = {})".format(
                self.embedding_file,
                example_embedding.shape
            ))
            self.embed_dim = example_embedding.shape[-1]

        self.cache_embeddings = cache_embeddings
        self.embedding_cache = {}
        if cache_embeddings:
            self._cache_all_embeddings()

    def _cache_all_embeddings(self):
        """
        Load all embeddings into memory for faster access.
        Warning: This can use significant RAM for large embedding files.
        """
        print("Caching embeddings in memory...")
        with h5py.File(self.embedding_file, "r") as f:
            for key in f.keys():
                self.embedding_cache[key] = f[key][:]
        print(f"Cached {len(self.embedding_cache)} embeddings in memory.")

    def convert(self, sample: MicrobiomeSample) -> Tuple[List[str], np.ndarray, np.ndarray]:
        # Get ASV data for this sample
        # Note: asv_id_subset should automatically filter out ASVs without a valid embedding stored in the h5 file.
        asv_ids, abunds = sample.relative_abundance_array(asv_id_subset=self.asv_id_subset)
        if len(asv_ids) == 0:
            raise Exception("No ASV found in the intersection between embedded ASV IDS and sample.")

        # Initialize tensors with padding
        n = len(asv_ids)
        features = np.zeros((n, self.embed_dim), dtype=float)

        # Fill in the tensors
        if self.cache_embeddings:
            # Use cached embeddings
            for asv_idx, asv_id in enumerate(asv_ids):
                features[asv_idx] = self.embedding_cache[asv_id]
        else:
            # Load embeddings from file
            with h5py.File(self.embedding_file, "r") as h5_file:
                for asv_idx, asv_id in enumerate(asv_ids):
                    features[asv_idx] = h5_file[asv_id][:]

        return asv_ids, features, abunds

    def get_abundance_profile_lightweight(self, sample: MicrobiomeSample) -> Tuple[List[str], np.ndarray]:
        """
        A smaller "lightweight" version of convert(), that only returns the subset list of ASV IDS with embeddings from the sample.
        :param sample:
        :return:
        """
        # Get ASV data for this sample
        # Note: asv_id_subset should automatically filter out ASVs without a valid embedding stored in the h5 file.
        asv_ids_subset, abunds_subset = sample.relative_abundance_array(asv_id_subset=self.asv_id_subset)
        return asv_ids_subset, abunds_subset / np.sum(abunds_subset)


# =========================================================================
class ASVDatasetForBaseline(Dataset):
    def __init__(
            self,
            sample_metadata: pd.DataFrame,
            abundance_table_dir: Path,
            asv_id_subset: Set[str],
            dtype: torch.dtype = torch.float32
    ):
        self.sample_df = sample_metadata
        self.dtype = dtype
        self.asv_id_subset = asv_id_subset
        self.sample_list = self.initialize_samples(abundance_table_dir)

    def initialize_samples(self, abundance_table_dir: Path) -> List[MicrobiomeSample]:
        """
        Create a mapping from dataset index to (project_id, sample_id, row_index).
        This allows efficient random access without groupby operations.
        """
        sample_list = []
        for proj_id, proj_section in self.sample_df.groupby("project"):
            proj = MicrobiomeProject(str(proj_id), abundance_table_dir, self.sample_df)
            target_sample_ids = set(proj_section['srs'])  # subset of samples from this project that we want to keep.
            for sample in proj.samples:
                if sample.sample_id not in target_sample_ids:
                    # only keep samples in the requested subset dataframe.
                    continue

                if len(sample.asv_ids.intersection(self.asv_id_subset)) == 0:
                    # only keep samples with at least one valid ASV (e.g. post-filter ASVs)
                    continue

                sample_list.append(sample)

        if len(sample_list) < self.sample_df.shape[0]:
            print(
                "Dataframe specified {} samples, but ended up with {} after ASV subset filtering.".format(self.sample_df.shape[0], len(sample_list))
            )

        # Finally, sort sample_list according to dataframe ordering.
        sample_ordering: Dict[str, int] = {
            str(row['srs']): row_idx
            for row_idx, (_, row) in enumerate(self.sample_df.iterrows())
        }
        sample_list = sorted(sample_list, key=lambda samp: sample_ordering[samp.sample_id])
        return sample_list

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> Tuple[str, List[str], Tensor]:
        sample = self.sample_list[idx]
        asv_id_ordering, abunds_subset = sample.relative_abundance_array(asv_id_subset=self.asv_id_subset)
        return sample.sample_id, asv_id_ordering, torch.from_numpy(abunds_subset).to(self.dtype)


class ASVPreembeddedDataset(Dataset):
    def __init__(
            self,
            sample_metadata: pd.DataFrame,
            abundance_table_dir: Path,
            embedding_h5_path: Path,
            dtype: torch.dtype = torch.float32
    ):
        self.sample_df = sample_metadata
        self.dtype = dtype
        self.sample_converter = MicrobiomeSampleEmbedding(embedding_h5_path, cache_embeddings=True)

        self.sample_list = self.initialize_samples(abundance_table_dir)

    def initialize_samples(self, abundance_table_dir: Path) -> List[MicrobiomeSample]:
        """
        Create a mapping from dataset index to (project_id, sample_id, row_index).
        This allows efficient random access without groupby operations.
        """
        sample_list = []
        for proj_id, proj_section in self.sample_df.groupby("project"):
            proj = MicrobiomeProject(str(proj_id), abundance_table_dir, self.sample_df)
            target_sample_ids = set(proj_section['srs'])  # subset of samples from this project that we want to keep.
            for sample in proj.samples:
                if sample.sample_id not in target_sample_ids:
                    # only keep samples in the requested subset dataframe.
                    continue

                if len(sample.asv_ids.intersection(self.sample_converter.asv_id_subset)) == 0:
                    # only keep samples with at least one valid ASV with embedding (e.g. post-filter ASVs)
                    continue

                sample_list.append(sample)

        if len(sample_list) < self.sample_df.shape[0]:
            print(
                "Dataframe specified {} samples, but ended up with {} after ASV subset filtering.".format(self.sample_df.shape[0], len(sample_list))
            )

        # Finally, sort sample_list according to dataframe ordering.
        sample_ordering: Dict[str, int] = {
            str(row['srs']): row_idx
            for row_idx, (_, row) in enumerate(self.sample_df.iterrows())
        }
        sample_list = sorted(sample_list, key=lambda samp: sample_ordering[samp.sample_id])
        return sample_list

    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor]:
        sample = self.sample_list[idx]
        asv_ids, features, abunds = self.sample_converter.convert(sample=sample)
        return sample.sample_id, torch.from_numpy(features).to(self.dtype), torch.from_numpy(abunds).to(self.dtype)

    def __len__(self) -> int:
        return len(self.sample_list)

    def embedding_dtype(self) -> torch.dtype:
        return self.dtype

    def embed_feature_dim(self) -> int:
        return self.sample_converter.embed_dim

    def true_abundance_profile(self, idx: int) -> Tuple[List[str], Tensor]:
        sample = self.sample_list[idx]
        asv_ids, abunds = sample.relative_abundance_array(asv_id_subset=self.sample_converter.asv_id_subset)
        return asv_ids, torch.from_numpy(abunds).to(self.dtype)


def collate_asv_profiles(
        batch: List[Tuple[str, Tensor, Tensor]]
) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
    batch_sz = len(batch)
    S_max = max(len(abunds) for _, _, abunds in batch)
    embed_dim = batch[0][1].shape[-1]
    feat_dtype = batch[0][1].dtype
    target_dtype = batch[0][2].dtype

    batch_sample_ids = []
    features_batch = torch.zeros((batch_sz, S_max, embed_dim), dtype=feat_dtype)
    asv_mask_batch = torch.zeros((batch_sz, S_max), dtype=torch.bool)
    targets_batch = torch.zeros((batch_sz, S_max), dtype=target_dtype)

    for b_idx, (sample_id, embeddings, targets) in enumerate(batch):
        S = len(targets)
        batch_sample_ids.append(sample_id)
        features_batch[b_idx, :S, :] = embeddings
        asv_mask_batch[b_idx, :S] = True
        targets_batch[b_idx, :S] = targets

    # sample_ids, training_batch_features, training_marker_mask, training_taxa_mask, training_y
    # features: (batch_sz, S_max, M, E), where M=1
    # marker_mask: (batch_sz, S_max, M), where M=1
    # taxa_mask: (batch_sz, S_max)
    # training_y: (batch_sz, S_max)
    return (
        batch_sample_ids,
        features_batch.unsqueeze(2),
        asv_mask_batch.unsqueeze(2),
        asv_mask_batch,
        targets_batch
    )


def create_dataloader(
        sample_df: pd.DataFrame,
        abundance_table_dir: Path,
        embedding_h5_path: Path,
        batch_size: int,
        num_workers: int,
        rng: Optional[torch.Generator],
        drop_last: bool,
        shuffle: bool,
        prefetch_factor: int = 2,
        dtype: torch.dtype = torch.float32
):
    dset = ASVPreembeddedDataset(
        sample_metadata=sample_df,
        abundance_table_dir=abundance_table_dir,
        embedding_h5_path=embedding_h5_path,
        dtype=dtype
    )
    return dset, DataLoader(
        dataset=dset,
        batch_size=batch_size, num_workers=num_workers,
        generator=rng, drop_last=drop_last, prefetch_factor=prefetch_factor,
        persistent_workers=True, pin_memory=True,
        shuffle=shuffle,
        collate_fn=collate_asv_profiles,
    )
