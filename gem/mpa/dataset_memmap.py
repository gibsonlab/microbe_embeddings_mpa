from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import *

from pathlib import Path
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from tensordict import TensorDict
from tqdm import tqdm

from .abundance_profile import MetaphlanProfile
from .dataset import MetaphlanDataset

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
        print("Performing memory-mapping allocation with {} threads.")
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
        n_tasks = 0
        for sample in dataset.samples:
            memmap_dir = cache_dir / sample.sample_id
            if (memmap_dir / "meta.json").exists():
                continue
            futures.append(
                executor.submit(allocate_sample, memmap_dir, sample, dataset)
            )
            n_tasks += 1

        # Process completed tasks with progress bar
        with tqdm(total=n_tasks, desc="Sample Allocation") as pbar:
            for future in as_completed(futures):
                _ = future.result()
                pbar.update(1)


class MetaphlanDatasetMemmapped(Dataset):
    """
    A class which pre-computes all tensors and stores into a memory-mapped tensordict.
    """
    def __init__(
            self,
            dataset_df: pd.DataFrame
    ):
        super().__init__()
        self.df = dataset_df
        self.tensor_cache: List[TensorDict] = []
        self.loaded = False

    def load_memmap_tensors(self, cache_dir: Path):
        print(f"Using tensor memmap directory: {cache_dir}")

        for sample_id, row in tqdm(self.df.iterrows()):
            memmap_dir = cache_dir / str(sample_id)
            if (memmap_dir / "meta.json").exists():
                # TensorDict is already allocated; load it from disk.
                x = TensorDict.load_memmap(memmap_dir)
            else:
                raise FileNotFoundError(f"Memory-mapped tensordict not found for sample: {sample_id}. Run perform_allocation() prior to load_memmap_tensors().")
            self.tensor_cache.append(x)
        self.loaded = True

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Load from pre-computed tensordict.
        """
        if not self.loaded:
            raise RuntimeError("Method load_memmap_tensors() must be run once prior to data access.")
        x = self.tensor_cache[idx]
        return x['targets'], x['mpadding'], x['spadding'], x['targets']
