from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import *

from pathlib import Path
import pandas as pd
import torch
from torch import Tensor
from tensordict import TensorDict
from tqdm import tqdm

from .abundance_profile import MetaphlanProfile
from .dataset import MetaphlanDataset, AbstractMetaphlanDataset

"""
From https://docs.pytorch.org/tensordict/main/saving.html

1) Saving a memmapped tensordict:
    x = TensorDict()
    x_disk = x.memmap("/path/to/saved/dir", num_threads=30)

2) Loading a memmapped tensordict:
    x = TensorDict.load_memmap("/path/to/saved/dir")
"""


def allocate_sample(memmap_dir: Path, sample: MetaphlanProfile, dataset: MetaphlanDataset) -> bool:
    """
    Allocate a single sample to memory-mapped storage.
    """
    assert not (memmap_dir / "meta.json").exists()
    memmap_dir.mkdir(exist_ok=True, parents=False)  # parent dir should already exist!
    _, features, marker_padding_mask, sgb_padding_mask, targets = dataset.load_sample_embeddings(sample)
    x = TensorDict()
    x['features'] = features
    x['mpadding'] = marker_padding_mask
    x['spadding'] = sgb_padding_mask
    x['targets'] = targets
    x.memmap(str(memmap_dir))
    return True


def perform_allocation(dataset: MetaphlanDataset, cache_dir: Path, num_threads: int):
    if num_threads <= 1:
        print("Performing memory-mapping allocation in single-threaded mode.")
        perform_allocation_single_thread(dataset, cache_dir)
    else:
        print(f"Performing memory-mapping allocation with {num_threads} threads.")
        perform_allocation_multi_thread(dataset, cache_dir, num_threads)


def perform_allocation_single_thread(dataset: MetaphlanDataset, cache_dir: Path):
    for sample in tqdm(dataset.samples, desc="Sample Allocation"):
        memmap_dir = cache_dir / sample.sample_id
        if (memmap_dir / "meta.json").exists():
            # TensorDict is already allocated; nothing to do.
            pass
        else:
            # Allocate the TensorDict.
            allocate_sample(memmap_dir, sample, dataset)


def perform_allocation_multi_thread(dataset: MetaphlanDataset, cache_dir: Path, num_threads: int):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = []
        futures_sample_id = dict()

        n_tasks = 0
        for sample in dataset.samples:
            memmap_dir = cache_dir / sample.sample_id
            if (memmap_dir / "meta.json").exists():
                continue

            future = executor.submit(allocate_sample, memmap_dir, sample, dataset)
            futures.append(future)
            futures_sample_id[future] = sample.sample_id
            n_tasks += 1

        # Process completed tasks with progress bar
        finished_states = dict()
        with tqdm(total=n_tasks, desc="Sample Allocation") as pbar:
            for future in as_completed(futures):
                try:
                    _ = future.result()
                    finished_states[futures_sample_id[future]] = (True, None)
                except Exception as e:
                    finished_states[futures_sample_id[future]] = (False, str(e))
                pbar.update(1)

    n_success = 0
    n_failed = 0
    for sample_id, (was_success, error_msg) in finished_states.items():
        if was_success:
            n_success += 1
        else:
            n_failed += 1
            print(f"Sample {sample_id} failed with error: {error_msg}")

    print("{} of {} allocation tasks successfully completed.".format(
        n_success, n_success + n_failed
    ))


class MetaphlanDatasetMemmapped(AbstractMetaphlanDataset):
    """
    A class which pre-computes all tensors and stores into a memory-mapped tensordict.
    """

    def __init__(
            self,
            sample_ids: List[str]
    ):
        super().__init__()
        self.tensor_cache: List[TensorDict] = []
        self.sample_ids: List[str] = sample_ids
        self.loaded = False

    def load_memmap_tensors(self, cache_dir: Path):
        print(f"Using tensor memmap directory: {cache_dir}")

        for sample_id in tqdm(self.sample_ids):
            memmap_dir = cache_dir / str(sample_id)
            if (memmap_dir / "meta.json").exists():
                # TensorDict is already allocated; load it from disk.
                x = TensorDict.load_memmap(memmap_dir)
            else:
                raise FileNotFoundError(f"Memory-mapped tensordict not found for sample: {sample_id}. Run perform_allocation() prior to load_memmap_tensors().")
            self.tensor_cache.append(x)
            self.sample_ids.append(str(sample_id))
        print("Finished loading memmapped tensors.")
        self.loaded = True

    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        """
        Load from pre-computed tensordict.
        """
        if not self.loaded:
            raise RuntimeError("Method load_memmap_tensors() must be run once prior to data access.")
        x = self.tensor_cache[idx]
        sample_id = self.sample_ids[idx]
        return sample_id, x['features'], x['mpadding'], x['spadding'], x['targets']

    def __len__(self) -> int:
        return len(self.tensor_cache)

    def embedding_dtype(self) -> torch.dtype:
        return self.tensor_cache[0]['features'].dtype

    def max_num_sgbs(self) -> int:
        return max(
            tdict['spadding'].sum().item()
            for tdict in self.tensor_cache
        )

    def max_num_markers(self) -> int:
        return max(  # max across all samples
            tdict['mpadding'].sum(dim=-1).max().item()  # max. # of markers among SGBs in sample
            for tdict in self.tensor_cache
        )

    def embed_feature_dim(self) -> int:
        return self.tensor_cache[0]['features'].shape[-1]
