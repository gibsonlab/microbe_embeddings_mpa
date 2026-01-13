from pathlib import Path
from typing import *
import pandas as pd

from tqdm import tqdm
import torch
from gem.mpa import MetaphlanHDF5Dataset, MetaphlanDatasetMemmapped, HDF5BatchShuffledSampler
from gem.ml.dataloader import MetaphlanDataLoader

import time
from contextlib import contextmanager

@contextmanager
def timer(name="Operation"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} took {end - start:.4f} seconds")


def initialize_test_dataset(dataset_tsv: Path):
    df = pd.read_csv(dataset_tsv, sep='\t')
    return df


def time_hdf5(hdf5_path: Path, n_iters: int = 100, batch_sz: int = 5):
    rng = torch.Generator()
    rng.manual_seed(12345)

    dset = MetaphlanHDF5Dataset(hdf5_path, model_dtype=torch.float32)
    dloader = MetaphlanDataLoader(
        dataset=dset,
        batch_size=batch_sz, num_workers=1, pin_memory=True,
        generator=rng, drop_last=False, prefetch_factor=2,
        persistent_workers=True,
        sampler=HDF5BatchShuffledSampler(dset, batch_sz, True),
    )
    with timer("HDF5"):
        for batch_idx, (training_sample_ids, _, _, _, _) in tqdm(enumerate(dloader), total=len(dloader)):
            # print(training_sample_ids)
            if batch_idx == n_iters - 1:
                break


def time_memmap(sample_ids: List[str], memmap_dir: Path, n_iters: int = 100, batch_sz: int = 5):
    dset = MetaphlanDatasetMemmapped(sample_ids=sample_ids)
    dset.load_memmap_tensors(memmap_dir)

    rng = torch.Generator()
    rng.manual_seed(12345)

    dloader = MetaphlanDataLoader(
        dataset=dset,
        batch_size=batch_sz, num_workers=1, pin_memory=True,
        generator=rng, drop_last=False, prefetch_factor=2,
        persistent_workers=True,
        shuffle=True,
    )
    with timer("Tensordict-memmap"):
        for batch_idx, (training_sample_ids, _, _, _, _) in tqdm(enumerate(dloader), total=len(dloader)):
            # print(training_sample_ids)
            if batch_idx == n_iters - 1:
                break


if __name__ == "__main__":
    df = initialize_test_dataset(Path("/data/cctm/youn/metaphlan_dset/model_training/test.tsv"))
    time_memmap(df['SampleID'].tolist(), Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmmap_samples"))
    # time_hdf5(Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/hdf5_samples/test.hdf5"))