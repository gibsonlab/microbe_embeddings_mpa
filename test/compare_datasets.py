from pathlib import Path
from typing import *
import pandas as pd

import torch
from gem.mpa import MetaphlanHDF5Dataset, MetaphlanDatasetMemmapped

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


def time_hdf5(hdf5_path: Path):
    dset = MetaphlanHDF5Dataset(hdf5_path, model_dtype=torch.float32)
    with timer("HDF5"):
        for i in range(len(dset)):
            _ = dset[i]


def time_memmap(sample_ids: List[str], memmap_dir: Path):
    dset = MetaphlanDatasetMemmapped(sample_ids=sample_ids)
    dset.load_memmap_tensors(memmap_dir)
    with timer("Tensordict-memmap"):
        for i in range(len(dset)):
            _ = dset[i]


if __name__ == "__main__":
    df = initialize_test_dataset(Path("/data/cctm/youn/metaphlan_dset/model_training/test.tsv"))
    time_hdf5(Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/hdf5_samples/test.hdf5"))
    time_memmap(df['SampleID'].tolist(), Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmmap_sample"))