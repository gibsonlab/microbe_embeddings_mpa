from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Sampler

from gem.datasets.mpa import AbstractMetaphlanPreembeddedDataset
from .collate import collate_fn_dynamic_alloc


def worker_init_fn(worker_id: int, base_rng_seed: int):
    """ Set random seed """
    import random, torch, numpy as np
    # import os  # debug
    # print(f"Initializing DataLoader worker ID {worker_id} (pid {os.getpid()})")  # debug
    worker_seed = base_rng_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ContiguousBatchSampler(Sampler):
    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool=True, seed: Optional[int]=None):
        super().__init__()
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        self.num_batches = dataset_size // batch_size

    def __iter__(self):
        # Create batch indices: [0, 1, 2, ...], [batch_sz, batch_sz+1, ...], ...
        batch_starts = list(range(0, self.num_batches * self.batch_size, self.batch_size))
        if self.shuffle:
            indices = torch.randperm(len(batch_starts), generator=self.rng).tolist()
            batch_starts = [batch_starts[i] for i in indices]
        for start in batch_starts:
            yield list(range(start, start + self.batch_size))

    def __len__(self) -> int:
        return self.num_batches


class MetaphlanDataLoader(DataLoader):
    def __init__(
            self,
            dataset: AbstractMetaphlanPreembeddedDataset,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            shuffle: bool = False,
            contiguous_batches: bool = False,
            drop_last: bool = False,
            collate_fn: Optional[Callable] = collate_fn_dynamic_alloc,
            worker_rng_seed: int = 31415,
            **dataloader_kwargs
    ):
        """
        Initialize the microbiome data loader.

        :param dataset: AbstractMetaphlanDataset object
        :param batch_size: Batch size
        :param shuffle: Whether to shuffle data
        :param num_workers: Number of worker processes
        :param pin_memory: Whether to pin memory
        :param dataloader_kwargs: Additional DataLoader arguments
        """
        if contiguous_batches:
            batch_sampler = ContiguousBatchSampler(len(dataset), batch_size=batch_size, shuffle=shuffle, seed=torch.initial_seed())
            super().__init__(
                dataset=dataset,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                batch_sampler=batch_sampler,
                # batch_size=batch_size,  # incompatible with batch_sampler option
                # shuffle=shuffle,  # incompatible with batch_sampler option
                # drop_last=drop_last,  # incompatible with batch_sampler option
                worker_init_fn=lambda wid: worker_init_fn(wid, worker_rng_seed),
                **dataloader_kwargs
            )
        else:
            super().__init__(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                shuffle=shuffle,
                drop_last=drop_last,
                worker_init_fn=lambda wid: worker_init_fn(wid, worker_rng_seed),
                **dataloader_kwargs
            )