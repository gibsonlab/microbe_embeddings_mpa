from pathlib import Path
from typing import *
import pandas as pd

from tqdm import tqdm
import torch
from gem.mpa import *
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
    print(f"Using memmap dir: {memmap_dir}")
    dset = MetaphlanDatasetMemmapped(sample_ids=sample_ids)
    dset.load_memmap_tensors(memmap_dir)

    rng = torch.Generator()
    rng.manual_seed(12345)

    from gem.ml import collate_fn_dynamic_alloc
    dloader = MetaphlanDataLoader(
        dataset=dset,
        batch_size=batch_sz, num_workers=2, pin_memory=True,
        generator=rng, drop_last=False, prefetch_factor=2,
        persistent_workers=True,
        shuffle=True,
        contiguous_batches=True,
        collate_fn=collate_fn_dynamic_alloc
    )
    with timer("Tensordict-memmap:dynamic"):
        for batch_idx, batch in tqdm(enumerate(dloader), total=len(dloader), desc="Batch-Load"):
            # x = batch[1].to("cuda").sum()
            # print("sum (cuda) = {}".format(x))
            # for sample in batch:
            #     print("{}: {}  --> sum = {}".format(sample[0], sample[1].shape, sample[1].sum().item()))
            if batch_idx == n_iters - 1:
                break


def time_memmap_padded(sample_ids: List[str], memmap_dir: Path, n_iters: int = 100, batch_sz: int = 5):
    print(f"Using memmap dir: {memmap_dir}")
    dset = MetaphlanDatasetMemmappedTensorDict(sample_ids=sample_ids)
    dset.load_memmap_tensors(memmap_dir)

    rng = torch.Generator()
    rng.manual_seed(12345)

    from gem.ml import collate_padded_tensordicts
    dloader = MetaphlanDataLoader(
        dataset=dset,
        batch_size=batch_sz, num_workers=2, pin_memory=True,
        generator=rng, drop_last=False, prefetch_factor=2,
        persistent_workers=True,
        shuffle=True,
        contiguous_batches=True,
        collate_fn=collate_padded_tensordicts
    )
    with timer("Tensordict-memmap:padded"):
        for batch_idx, batch in tqdm(enumerate(dloader), total=len(dloader), desc="Batch-Load"):
            x = batch[1].to("cuda", non_blocking=True).sum()
            # print("sum (cuda) = {}".format(x))
            # for sample in batch:
            #     print("{}: {}  --> sum = {}".format(sample[0], sample[1].shape, sample[1].sum().item()))
            if batch_idx == n_iters - 1:
                break


def time_memmap_large(sample_ids: List[str], memmap_dir: Path, n_iters: int = 100, batch_sz: int = 5):
    print(f"Using memmap dir: {memmap_dir}")
    dset = MetaphlanDatasetMemmappedLarge(memmap_dir=memmap_dir, assume_contiguous_access=True)
    assert len(dset.sample_ids) == len(sample_ids)

    rng = torch.Generator()
    rng.manual_seed(12345)

    dloader = MetaphlanDataLoader(
        dataset=dset,
        batch_size=batch_sz, num_workers=2, pin_memory=False,
        generator=rng, drop_last=False, prefetch_factor=2,
        persistent_workers=True,
        shuffle=True,
        contiguous_batches=True,
        collate_fn=lambda x: x,
        # this doesn't need a collate_fn, since dset implements __getitems__, supported by torch 1.12 onwards
    )
    # preload
    with timer("Tensordict-memmap:large"):
        for batch_idx, batch in tqdm(enumerate(dloader), total=len(dloader), desc="Batch-Load"):
            x = batch[1].to("cuda", non_blocking=True).sum()
            if batch_idx == n_iters - 1:
                break


if __name__ == "__main__":
    dset_name = "test"
    df = initialize_test_dataset(Path(f"/data/cctm/youn/metaphlan_dset/model_training/{dset_name}.tsv"))

    #time_memmap(df['SampleID'].tolist(), Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_padded"))
    ### the padded version has a bug --- TensorDict uses padding_1d built-in function, which can't stack tensors of shape (L_i, M, D) for matching M,D.
    #time_memmap_padded(df['SampleID'].tolist(), Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_padded"))
    time_memmap_large(df['SampleID'].tolist(), Path(f"/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_complete_large/evo/{dset_name}"))
    # time_hdf5(Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/hdf5_samples/test.hdf5"))