from typing import *
import json

from pathlib import Path
import torch
from torch import Tensor
from tensordict import TensorDict, MemoryMappedTensor
from tqdm import tqdm

from .dataset import AbstractMetaphlanDataset


def parse_torch_dtype(s: str) -> torch.dtype:
    # allow both "float32" and "torch.float32"
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    dt = getattr(torch, s)
    if not isinstance(dt, torch.dtype):
        raise TypeError(f"{s} is not a torch.dtype")
    return dt


def parse_sample_ids_memmap(memmap_dir: Path) -> List[str]:
    with open(memmap_dir / "sample_ids.txt", "rt") as f:
        cached_sample_ids = [line.strip() for line in f]
    return cached_sample_ids


def parse_tdict_metadata(memmap_dir: Path) -> Tuple[int, int, int, int, torch.dtype]:
    with open(memmap_dir / "meta.json", "rt") as f:
        metadata = json.load(f)
        return int(metadata["N"]), int(metadata["S"]), int(metadata["M"]), int(metadata["E"]), parse_torch_dtype(metadata['dtype'])


def allocate_big_memmap_tdict(
    sample_ids: List[str],
    out_dir: Path,
    S_max_global: int,
    M_max_global: int,
    embed_dim: int,
    dtype: torch.dtype = torch.float32,
) -> TensorDict:
    """
    Allocates a big memmapped tensordict, with the requested shape and dtype.

    :param sample_ids:
    :param out_dir:
    :param S_max_global:
    :param M_max_global:
    :param embed_dim:
    :param dtype:
    :return:
    """
    if dtype == torch.float32:
        dtype_str = "torch.float32"
    else:
        raise ValueError(f"This script currently does not support the dtype {dtype}")

    print(f"Target allocation path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(sample_ids)

    # 1) Allocate on-disk MemoryMappedTensors
    print("Feature shape: ({}, {}, {}, {})".format(N, S_max_global, M_max_global, embed_dim))
    big_features = MemoryMappedTensor.empty(
        (N, S_max_global, M_max_global, embed_dim),
        dtype=dtype,
        filename=str(out_dir / "features.mmap"),
    )
    big_mpadding = MemoryMappedTensor.empty(
        (N, S_max_global, M_max_global),
        dtype=torch.bool,
        filename=str(out_dir / "mpadding.mmap"),
    )
    big_spadding = MemoryMappedTensor.empty(
        (N, S_max_global),
        dtype=torch.bool,
        filename=str(out_dir / "spadding.mmap"),
    )
    big_targets = MemoryMappedTensor.empty(
        (N, S_max_global),
        dtype=dtype,
        filename=str(out_dir / "targets.mmap"),
    )

    big_td = TensorDict(
        {
            "features": big_features,
            "mpadding": big_mpadding,
            "spadding": big_spadding,
            "targets": big_targets,
        },
        batch_size=[N],
        device="cpu",
    )

    with open(out_dir / "sample_ids.txt", "wt") as f:
        for s_id in sample_ids:
            print(s_id, file=f)

    with open(out_dir / "meta.json", "wt") as f:
        json.dump(
            {
                "N": N,
                "S": S_max_global,
                "M": M_max_global,
                "E": embed_dim,
                "dtype": dtype_str
            },
            f
        )

    return big_td


def fetch_preallocated_tdict(memmap_dir: Path) -> Tuple[TensorDict, int, int, int, torch.dtype]:
    assert memmap_dir.is_dir()
    assert memmap_dir.exists()
    N, S_max, M_max, E, dtype = parse_tdict_metadata(memmap_dir)
    print("Fetching pre-allocated tensordict. Features = ({}, {}, {}, {}), dtype = {}".format(
        N, S_max, M_max, E,
        dtype
    ))
    features = MemoryMappedTensor.from_filename(
        filename=str(memmap_dir / "features.mmap"),
        dtype=dtype,
        shape=torch.Size((N, S_max, M_max, E)),
    )
    mpadding = MemoryMappedTensor.from_filename(
        filename=str(memmap_dir / "mpadding.mmap"),
        dtype=torch.bool,
        shape=torch.Size((N, S_max, M_max)),
    )
    spadding = MemoryMappedTensor.from_filename(
        filename=str(memmap_dir / "spadding.mmap"),
        dtype=torch.bool,
        shape=torch.Size((N, S_max)),
    )
    targets = MemoryMappedTensor.from_filename(
        filename=str(memmap_dir / "targets.mmap"),
        dtype=dtype,
        shape=torch.Size((N, S_max)),
    )
    return TensorDict(
        {"features": features, "mpadding": mpadding, "spadding": spadding, "targets": targets},
        batch_size=[N],
    ), S_max, M_max, E, dtype


class MetaphlanDatasetMemmappedLarge(AbstractMetaphlanDataset):
    """
    A class which pre-computes all tensors and stores into a memory-mapped tensordict.
    """

    def __init__(self, memmap_dir: Path, assume_contiguous_access: bool):
        super().__init__()
        self.sample_ids: List[str] = parse_sample_ids_memmap(memmap_dir)
        self.tensordict, self.max_sgbs, self.max_markers, self.embed_dim, self.dtype = fetch_preallocated_tdict(memmap_dir)
        self.assume_contiguous_access = assume_contiguous_access  # for use with ContiguousBatchSampler

    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        """
        Load from pre-computed tensordict.
        """
        return (
            self.sample_ids[idx],
            self.tensordict['features'][idx], self.tensordict['mpadding'][idx],
            self.tensordict['spadding'][idx], self.tensordict['targets'][idx]
        )

    def __getitems__(self, indices: List[int]) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
        """
        Load from pre-computed tensordict, in a batch.
        Supported automatically by torch 1.12 onwards in DataLoader.
        :param indices:
        :return:
        """
        if self.assume_contiguous_access:
            start_idx = indices[0]
            end_idx = indices[-1] + 1
            return (
                [self.sample_ids[i] for i in indices],
                self.tensordict['features'][start_idx:end_idx], self.tensordict['mpadding'][start_idx:end_idx],
                self.tensordict['spadding'][start_idx:end_idx], self.tensordict['targets'][start_idx:end_idx]
            )
        else:
            return (
                [self.sample_ids[i] for i in indices],
                self.tensordict['features'][indices], self.tensordict['mpadding'][indices],
                self.tensordict['spadding'][indices], self.tensordict['targets'][indices]
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def embedding_dtype(self) -> torch.dtype:
        return self.dtype

    def max_num_sgbs(self) -> int:
        return self.max_sgbs

    def max_num_markers(self) -> int:
        return self.max_markers

    def embed_feature_dim(self) -> int:
        return self.embed_dim

    def true_abundance_profile(self, idx: int) -> Tensor:
        return self.tensordict['targets'][idx]
