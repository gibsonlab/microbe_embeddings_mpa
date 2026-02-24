from typing import Tuple, List
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import AbstractMetaphlanPreembeddedDataset
from .. import MetaphlanProfileParser


def get_meta_file(fpath: Path) -> Path:
    tokens = fpath.name.split(".")
    print(tokens)
    if tokens[-2].startswith("ipca"):
        # strip the ipca_<dim> token, if present.
        basename = ".".join(tokens[:-2])
    else:
        basename = ".".join(tokens[:-1])
    return fpath.parent / f"{basename}.meta"


class TorchStackedMetaphlanPreembeddedDataset(AbstractMetaphlanPreembeddedDataset):
    def __init__(
            self,
            dataset_df: pd.DataFrame,
            file_path: Path,
            dtype: torch.dtype = torch.float32,
    ):
        self.df = dataset_df
        self.samples = list(MetaphlanProfileParser(dataset_df).samples())
        self.dtype = dtype

        print("Target torch embedding array: {}".format(file_path))
        meta_path = get_meta_file(file_path)
        assert meta_path.exists(), f"Metadata file for torch embedding file {file_path.name} not found!"
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

        self.embeddings = torch.load(file_path)
        print("Loaded embedding tensor of shape: {}".format(self.embeddings.shape))
        self.all_marker_masks = ~(torch.isnan(self.embeddings).any(dim=-1))
        self.embeddings[torch.isnan(self.embeddings)] = 0.0

        with open(file_path.with_suffix(".sgb.txt"), "rt") as f:
            self.sgb_order = [l.strip() for l in f if len(l.strip()) > 0]
        assert len(self.sgb_order) == self.embeddings.shape[0], "SGB id length ({}) does not match embeddings shape ({})".format(
            len(self.sgb_order), self.embeddings.shape[0]
        )
        self.sgb_indices = {s_id: i for i, s_id in enumerate(self.sgb_order)}

    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        sample = self.samples[idx]
        n_sgbs = len(sample.taxa_ids)
        n_markers = self.embeddings.shape[1]
        embed_dim = self.embeddings.shape[2]

        # Pre-allocate tensors on target device
        features = torch.zeros((n_sgbs, n_markers, embed_dim), dtype=self.dtype)
        marker_mask = torch.zeros((n_sgbs, n_markers), dtype=torch.bool)

        for sgb_idx, sgb_id in enumerate(sample.taxa_ids):
            if sgb_id not in self.sgbs_without_embedding:
                arr_idx = self.sgb_indices[sgb_id]
                features[sgb_idx] = self.embeddings[arr_idx]
                marker_mask[sgb_idx] = self.all_marker_masks[arr_idx]
            else:
                # leave the features/marker mask at zero (sgb mask is still True, so this asks models to still produce a guess.)
                pass

        sgb_mask = torch.ones(n_sgbs, dtype=torch.bool)
        targets = torch.from_numpy(sample.abundances_ensure_normalized).to(self.dtype)
        return sample.sample_id, features, marker_mask, sgb_mask, targets

    def __getitems__(self, indices: List[int]) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
        batch_sz = len(indices)
        samples = [self.samples[i] for i in indices]
        max_sgbs = max(len(sample.taxa_ids) for sample in samples)
        max_markers = self.embeddings.shape[1]
        embed_dim = self.embeddings.shape[2]

        sample_ids = [s.sample_id for s in samples]
        features = torch.zeros((batch_sz, max_sgbs, max_markers, embed_dim), dtype=self.dtype)
        marker_mask = torch.zeros((batch_sz, max_sgbs, max_markers), dtype=torch.bool)
        sgb_mask = torch.zeros((batch_sz, max_sgbs), dtype=torch.bool)
        targets = torch.zeros((batch_sz, max_sgbs), dtype=self.dtype)

        # Collect all array indices needed across the whole batch
        needed_arr_indices = set()
        for sample in samples:
            for sgb_id in sample.taxa_ids:
                if sgb_id not in self.sgbs_without_embedding and sgb_id in self.sgb_indices:
                    needed_arr_indices.add(self.sgb_indices[sgb_id])

        for sample_idx, sample in enumerate(samples):
            for sgb_idx, sgb_id in enumerate(sample.taxa_ids):
                if sgb_id not in self.sgbs_without_embedding and sgb_id in self.sgb_indices:
                    sgb_arr_idx = self.sgb_indices[sgb_id]
                    features[sample_idx, sgb_idx] = self.embeddings[sgb_arr_idx]
                    marker_mask[sample_idx, sgb_idx] = self.all_marker_masks[sgb_arr_idx]

            sgb_mask[sample_idx, :len(sample.taxa_ids)] = True
            targets[sample_idx, :len(sample.taxa_ids)] = torch.from_numpy(sample.abundances_ensure_normalized).to(self.dtype)

        return sample_ids, features, marker_mask, sgb_mask, targets

    def __len__(self) -> int:
        return len(self.samples)

    def embedding_dtype(self) -> torch.dtype:
        return self.dtype

    def max_num_sgbs(self) -> int:
        return max(len(sample.taxa_ids) for sample in self.samples)

    def max_num_markers(self) -> int:
        return self.embeddings.shape[1]

    def embed_feature_dim(self) -> int:
        return self.embeddings.shape[-1]

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

