from typing import Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import AbstractMetaphlanPreembeddedDataset
from .. import MetaphlanProfileParser


class NumpyMemmappedMetaphlanPreembeddedDataset(AbstractMetaphlanPreembeddedDataset):
    def __init__(
            self,
            dataset_df: pd.DataFrame,
            file_path: Path,
            dtype: torch.dtype = torch.float32,
    ):
        self.df = dataset_df
        self.samples = list(MetaphlanProfileParser(dataset_df).samples())
        self.dtype = dtype

        print("target numpy memmap array: {}".format(file_path))
        meta_path = file_path.with_suffix(".meta")
        assert meta_path.exists(), f"Metadata file for memmap embedding file {file_path.name} not found!"
        with open(meta_path, "rt") as f:
            self.sgbs_without_embedding = set()
            _dtype = f.readline().strip()
            _shape = tuple(map(int, f.readline().split(',')))
            missing_line = f.readline().strip()
            assert missing_line.startswith("MISSING="), "Invalid line format. got: {}".format(missing_line)
            missing_ct = int(missing_line.split("=")[1])
            for _ in range(missing_ct):
                s_id = f.readline().strip()
                assert s_id.startswith("SGB")
                self.sgbs_without_embedding.add(s_id)

        self.array = np.memmap(
            file_path,
            dtype=_dtype,
            mode='r',
            shape=_shape,
        )
        self.all_marker_masks = ~(np.isnan(self.array).any(axis=-1))

        with open(file_path.with_suffix(".sgb.txt"), "rt") as f:
            self.sgb_order = [l.strip() for l in f if len(l.strip()) > 0]
        assert len(self.sgb_order) == self.array.shape[0], "SGB id length ({}) does not match memmap shape ({})".format(
            len(self.sgb_order), self.array.shape[0]
        )
        self.sgb_indices = {s_id: i for i, s_id in enumerate(self.sgb_order)}

    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        sample = self.samples[idx]
        n_sgbs = len(sample.taxa_ids)
        n_markers = self.array.shape[1]
        embed_dim = self.array.shape[2]

        # Pre-allocate tensors on target device
        features = torch.zeros((n_sgbs, n_markers, embed_dim), dtype=self.dtype)
        marker_mask = torch.zeros((n_sgbs, n_markers), dtype=torch.bool)

        for sgb_idx, sgb_id in enumerate(sample.taxa_ids):
            if sgb_id not in self.sgbs_without_embedding:
                arr_idx = self.sgb_indices[sgb_id]
                embed_slice = self.array[arr_idx]
                features[sgb_idx] = torch.from_numpy(embed_slice)
                marker_mask[sgb_idx] = torch.from_numpy(
                    ~np.isnan(embed_slice).any(axis=-1)
                )
            else:
                # leave the features/marker mask at zero (sgb mask is still True, so this asks models to still produce a guess.)
                pass

        sgb_mask = torch.ones(n_sgbs, dtype=torch.bool)
        targets = torch.from_numpy(sample.abundances_ensure_normalized).to(self.dtype)
        features[torch.isnan(features)] = 0.0
        return sample.sample_id, features, marker_mask, sgb_mask, targets

    def __getitems__(self, indices: List[int]) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
        batch_sz = len(indices)
        samples = [self.samples[i] for i in indices]
        max_sgbs = max(len(sample.taxa_ids) for sample in samples)
        max_markers = self.array.shape[1]
        embed_dim = self.array.shape[2]

        sample_ids = [s.sample_id for s in samples]
        marker_mask = torch.zeros((batch_sz, max_sgbs, max_markers), dtype=torch.bool)
        sgb_mask = torch.zeros((batch_sz, max_sgbs), dtype=torch.bool)
        targets = torch.zeros((batch_sz, max_sgbs), dtype=self.dtype)

        # Collect all array indices needed across the whole batch
        needed_arr_indices = set()
        for sample in samples:
            for sgb_id in sample.taxa_ids:
                if sgb_id not in self.sgbs_without_embedding and sgb_id in self.sgb_indices:
                    needed_arr_indices.add(self.sgb_indices[sgb_id])

        # Single bulk read from memmap, store in RAM.
        needed_arr_indices = sorted(needed_arr_indices)
        assert len(needed_arr_indices) > 0, "Each __getitems__ call must invoke at least one memmap index!"
        bulk = self.array[needed_arr_indices]  # one read, shape: (N, markers, embed_dim)
        bulk_map = {arr_idx: i for i, arr_idx in enumerate(needed_arr_indices)}

        # build a numpy array first, then convert to tensor once
        features_np = np.zeros((batch_sz, max_sgbs, max_markers, embed_dim), dtype=np.float32)
        for sample_idx, sample in enumerate(samples):
            for sgb_idx, sgb_id in enumerate(sample.taxa_ids):
                if sgb_id not in self.sgbs_without_embedding and sgb_id in self.sgb_indices:
                    features_np[sample_idx, sgb_idx] = bulk[bulk_map[self.sgb_indices[sgb_id]]]

            sgb_mask[sample_idx, :len(sample.taxa_ids)] = True
            targets[sample_idx, :len(sample.taxa_ids)] = torch.from_numpy(
                sample.abundances_ensure_normalized
            ).to(self.dtype)

        features_np[np.isnan(features_np)] = 0.0
        features = torch.from_numpy(features_np)

        assert features.shape == torch.Size([batch_sz, max_sgbs, max_markers, embed_dim])


        # for sample_idx, sample in enumerate(samples):
        #     for sgb_idx, sgb_id in enumerate(sample.taxa_ids):
        #         if sgb_id not in self.sgbs_without_embedding and sgb_id in self.sgb_indices:
        #             sgb_arr_idx = self.sgb_indices[sgb_id]
        #             embed_slice = bulk[bulk_map[sgb_arr_idx]]
        #             features[sample_idx, sgb_idx] = torch.from_numpy(embed_slice).to(dtype=self.dtype)
        #             marker_mask[sample_idx, sgb_idx] = torch.from_numpy(self.all_marker_masks[sgb_arr_idx])
        #
        #     sgb_mask[sample_idx, :len(sample.taxa_ids)] = True
        #     targets[sample_idx, :len(sample.taxa_ids)] = torch.from_numpy(
        #         sample.abundances_ensure_normalized
        #     ).to(self.dtype)

        # features[torch.isnan(features)] = 0.0
        return sample_ids, features, marker_mask, sgb_mask, targets

    def __len__(self) -> int:
        return len(self.samples)

    def embedding_dtype(self) -> torch.dtype:
        return self.dtype

    def max_num_sgbs(self) -> int:
        return max(len(sample.taxa_ids) for sample in self.samples)

    def max_num_markers(self) -> int:
        return self.array.shape[1]

    def embed_feature_dim(self) -> int:
        return self.array.shape[-1]

    def true_abundance_profile(self, idx: int) -> Tensor:
        sample = self.samples[idx]
        sgb_ids = sample.taxa_ids
        num_sgbs = len(sgb_ids)
        targets = torch.zeros(num_sgbs, dtype=self.dtype)
        for i, (sgb_id, sgb_abund) in enumerate(zip(sample.taxa_ids, sample.abundances)):
            if sgb_id in self.sgb_indices:
                targets[i] = sgb_abund

        targets = targets / targets.sum()
        return targets

    def create_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=self,
            collate_fn=lambda batch: batch,
            **kwargs
        )

