from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import *

from pathlib import Path
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
def add_marker_padding(x: Tensor, n_markers: int) -> Tensor:
    """

    :param x: A tensor of shape (S, m, *), where * is potentially empty.
    :param n_markers: An integer M >= m.
    :return:
    """
    assert x.shape[1] <= n_markers, "Can't pad a tensor with marker dimension {} into a tensor of shape {}".format(
        x.shape[1], n_markers
    )

    if len(x.shape) == 2:
        new_x = torch.zeros(size=(x.shape[0], n_markers), dtype=x.dtype)
    else:
        new_x = torch.zeros(size=(x.shape[0], n_markers, *x.shape[2:]), dtype=x.dtype)

    m = x.shape[1]
    new_x[:, :m] = x
    return new_x


def allocate_sample(memmap_dir: Path, sample: MetaphlanProfile, max_num_markers: int, dataset: MetaphlanDataset) -> bool:
    """
    Allocate a single sample to memory-mapped storage.
    """
    assert not (memmap_dir / "meta.json").exists()
    memmap_dir.mkdir(exist_ok=True, parents=False)  # parent dir should already exist!
    _, features, marker_padding_mask, sgb_padding_mask, targets = dataset.load_sample_embeddings(sample)

    marker_padding_mask = add_marker_padding(marker_padding_mask, max_num_markers)
    features = add_marker_padding(features, max_num_markers)

    x = TensorDict()
    x['features'] = features
    x['mpadding'] = marker_padding_mask
    x['spadding'] = sgb_padding_mask
    x['targets'] = targets
    x.memmap(str(memmap_dir))
    return True


def perform_allocation(dataset: MetaphlanDataset, cache_dir: Path, num_threads: int, max_num_markers: int):
    if num_threads <= 1:
        print("Performing memory-mapping allocation in single-threaded mode.")
        perform_allocation_single_thread(dataset, cache_dir, max_num_markers)
    else:
        print(f"Performing memory-mapping allocation with {num_threads} threads.")
        perform_allocation_multi_thread(dataset, cache_dir, num_threads, max_num_markers)


def perform_allocation_single_thread(dataset: MetaphlanDataset, cache_dir: Path, max_num_markers: int):
    for sample in tqdm(dataset.samples, desc="Sample Allocation"):
        memmap_dir = cache_dir / sample.sample_id
        if (memmap_dir / "meta.json").exists():
            # TensorDict is already allocated; nothing to do.
            pass
        else:
            # Allocate the TensorDict.
            allocate_sample(memmap_dir, sample, max_num_markers, dataset)


def perform_allocation_multi_thread(dataset: MetaphlanDataset, cache_dir: Path, num_threads: int, max_num_markers: int):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = []
        futures_sample_id = dict()

        n_tasks = 0
        for sample in dataset.samples:
            memmap_dir = cache_dir / sample.sample_id
            if (memmap_dir / "meta.json").exists():
                continue

            future = executor.submit(allocate_sample, memmap_dir, sample, max_num_markers, dataset)
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

        for sample_id in tqdm(self.sample_ids, desc="Load-Memmap"):
            memmap_dir = cache_dir / str(sample_id)
            if (memmap_dir / "meta.json").exists():
                # TensorDict is already allocated; load it from disk.
                x = TensorDict.load_memmap(memmap_dir)
            else:
                raise FileNotFoundError(f"Memory-mapped tensordict not found for sample: {sample_id}. Run perform_allocation() prior to load_memmap_tensors().")
            self.tensor_cache.append(x)
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
        if not self.loaded:
            raise RuntimeError("Method load_memmap_tensors() must be run once prior to data access.")
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

    def true_abundance_profile(self, idx: int) -> Tensor:
        _, _, _, _, abunds = self.__getitem__(idx)
        return abunds


class MetaphlanDatasetMemmappedTensorDict(AbstractMetaphlanDataset):
    """
    A re-implementation of MetaphlanDatasetMemmapped.
    Instead of returning the memmapped tensors by unpacking the dictionary, it returns the raw tensordict object
    per sample instead.
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

        for sample_id in tqdm(self.sample_ids, desc="Load-Memmap"):
            memmap_dir = cache_dir / str(sample_id)
            if (memmap_dir / "meta.json").exists():
                # TensorDict is already allocated; load it from disk.
                x = TensorDict.load_memmap(memmap_dir)
            else:
                raise FileNotFoundError(f"Memory-mapped tensordict not found for sample: {sample_id}. Run perform_allocation() prior to load_memmap_tensors().")
            self.tensor_cache.append(x)
        print("Finished loading memmapped tensors.")
        self.loaded = True

    def __getitem__(self, idx: int) -> Tuple[str, TensorDict]:
        return self.sample_ids[idx], self.tensor_cache[idx]

    def __len__(self) -> int:
        return len(self.smaple_ids)

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

    def true_abundance_profile(self, idx: int) -> Tensor:
        return self.tensor_cache[idx]['targets']
