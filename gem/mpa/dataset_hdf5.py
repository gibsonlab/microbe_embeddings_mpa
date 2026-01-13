from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import h5py
import torch
from torch import Tensor

from .dataset import AbstractMetaphlanDataset


class MetaphlanHDF5Dataset(AbstractMetaphlanDataset):
    def __init__(
            self,
            hdf5_path: Path,
            model_dtype: torch.dtype,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path

        # Open file once to get metadata
        with h5py.File(hdf5_path, 'r') as f:
            self.n_samples = f['features'].shape[0]
            self.sample_ids = [s.decode() for s in f['sample_ids'][:]]

        # Create batch indices
        self.model_dtype = model_dtype
        self._file = None

    def _get_file_handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
        return self._file

    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        f = self._get_file_handle()

        features = torch.from_numpy(f['features'][idx]).to(dtype=self.model_dtype)
        mpadding = torch.from_numpy(f['mpadding'][idx]).to(dtype=torch.bool)
        spadding = torch.from_numpy(f['spadding'][idx]).to(dtype=torch.bool)
        targets = torch.from_numpy(f['targets'][idx]).to(dtype=self.model_dtype)
        sample_id = self.sample_ids[idx]

        return sample_id, features, mpadding, spadding, targets

    def __len__(self) -> int:
        return self.n_samples

    def embedding_dtype(self) -> torch.dtype:
        return self.model_dtype

    def max_num_sgbs(self) -> int:
        f = self._get_file_handle()
        return max(
            int(f['spadding'][i].sum())
            for i in range(self.n_samples)
        )

    def max_num_markers(self) -> int:
        f = self._get_file_handle()
        return max(  # max across all samples
            int(f['mpadding'][i].sum(dim=-1).max())  # max. # of markers among SGBs in sample
            for i in range(self.n_samples)
        )

    def embed_feature_dim(self) -> int:
        f = self._get_file_handle()
        return f['features'][0].shape[-1]

    def true_abundance_profile(self, idx: int) -> Tensor:
        f = self._get_file_handle()
        return torch.from_numpy(f['targets'][idx]).to(dtype=self.model_dtype)

    def __del__(self):
        if self._file is not None:
            self._file.close()


class HDF5BatchShuffledSampler(torch.utils.data.Sampler):
    """
    Custom sampler that ensures we iterate through samples in batch-shuffled order.
    This is cleaner than relying on __getitem__ mapping.
    """
    def __init__(
            self,
            data_source: MetaphlanHDF5Dataset,
            batch_size: int,
            shuffle: bool = True,
            rng_seed: Optional[int] = None,
    ):
        super().__init__()
        self.data_source = data_source
        self.shuffle = shuffle
        self.n_samples = len(data_source)
        self.batch_size = batch_size
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self.rng = np.random.RandomState(rng_seed)

    def __iter__(self):
        # Use epoch-based seeding for reproducibility
        if self.shuffle:
            batch_order = self.rng.permutation(self.n_batches).tolist()
        else:
            batch_order = list(range(self.n_batches))

        # Generate indices in batch-shuffled order
        indices = []
        for batch_idx in batch_order:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            indices.extend(range(start_idx, end_idx))

        return iter(indices)

    def __len__(self):
        return self.n_samples
