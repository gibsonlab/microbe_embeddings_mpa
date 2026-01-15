import argparse
from pathlib import Path
from typing import *
from tqdm import tqdm
import json

import pandas as pd
import torch
from tensordict import TensorDict, MemoryMappedTensor

from gem.mpa import MetaphlanDataset, MetaphlanMarkerEmbedding


def populate_memmap_tdict(
        dataset: MetaphlanDataset,
        tdict_start_idx: int,
        big_tdict: TensorDict
):
    for sample_idx, sample in enumerate(tqdm(dataset.samples, desc="Sample Allocation")):
        _, features, marker_padding_mask, sgb_padding_mask, targets = dataset.load_sample_embeddings(sample)

        S, M = features.shape[:2]
        big_idx = tdict_start_idx + sample_idx
        big_tdict['features'][big_idx, :S, :M, :] = features
        big_tdict['mpadding'][big_idx, :S, :M] = marker_padding_mask
        big_tdict['spadding'][big_idx, :S] = sgb_padding_mask
        big_tdict['targets'][big_idx, :S] = targets


def parse_torch_dtype(s: str) -> torch.dtype:
    # allow both "float32" and "torch.float32"
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    dt = getattr(torch, s)
    if not isinstance(dt, torch.dtype):
        raise TypeError(f"{s} is not a torch.dtype")
    return dt


def parse_tdict_metadata(memmap_dir: Path) -> Tuple[int, int, int, int, torch.dtype]:
    with open(memmap_dir / "meta.json", "rt") as f:
        metadata = json.load(f)
        return int(metadata["N"]), int(metadata["S"]), int(metadata["M"]), int(metadata["E"]), parse_torch_dtype(metadata['dtype'])


def fetch_preallocated_tdict(memmap_dir: Path) -> TensorDict:
    assert memmap_dir.is_dir()
    assert memmap_dir.exists()
    N, S_max, M_max, E, dtype = parse_tdict_metadata(memmap_dir)
    features = MemoryMappedTensor.from_filename(
        filename=str(memmap_dir / "features.mmap"),
        dtype=dtype,
        shape=torch.Size(N, S_max, M_max, E),
    )
    mpadding = MemoryMappedTensor.from_filename(
        filename=str(memmap_dir / "mpadding.mmap"),
        dtype=torch.bool,
        shape=torch.Size(N, S_max, M_max),
    )
    spadding = MemoryMappedTensor.from_filename(
        filename=str(memmap_dir / "spadding.mmap"),
        dtype=torch.bool,
        shape=torch.Size(N, S_max),
    )
    targets = MemoryMappedTensor.from_filename(
        filename=str(memmap_dir / "targets.mmap"),
        dtype=dtype,
        shape=torch.Size(N, S_max),
    )
    return TensorDict(
        {"features": features, "mpadding": mpadding, "spadding": spadding, "targets": targets},
        batch_size=[N],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-tsv", dest="dataset_tsv", required=True, type=str)
    parser.add_argument("-e", "--embedding-dir", dest="marker_embedding_basedir", required=True, type=str)
    parser.add_argument("-m", "--memmap-dir", dest="memmap_dir", required=True, type=str)
    parser.add_argument("--start", dest="start_row", required=False, type=int, default=0)  # inclusive
    parser.add_argument("--end", dest="end_row", required=False, type=int, default=-1)  # inclusive
    parser.add_argument("--dimension-reduce", dest="dimension_reduce_pca", required=False, default=None, type=int,
                        help="If specified (an integer greater than zero), will perform incremental PCA on the entire"
                             "set of embeddings for dimensionality reduction.")
    parser.add_argument("--pca-batch-size", dest="ipca_batch_size", required=False, default=10000, type=int,
                        help="Specify the batch size for incremental PCA. Default: 10000")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    memmap_dir = Path(args.memmap_dir)

    dataset_df_full = pd.read_csv(args.dataset_tsv, sep='\t', index_col="SampleID")
    start_idx = args.start_row - 1
    if args.end_row == -1:
        print(f"Populating row #{args.start_row} onwards.")
        dataset_df = dataset_df_full.iloc[start_idx:]
    else:
        print(f"Populating row #{args.start_row} ~ #{args.end_row} (inclusive).")
        end_idx = args.end_row
        dataset_df = dataset_df_full.iloc[start_idx:end_idx]

    marker_embedding = MetaphlanMarkerEmbedding(
        marker_embedding_basedir=Path(args.marker_embedding_basedir),
        dimension_reduce_pca=args.dimension_reduce_pca,
        ipca_batch_size=args.ipca_batch_size,
    )
    dataset = MetaphlanDataset(dataset_df, marker_embedding)

    # validate sample ID ordering.
    print("Validating sample ordering.")
    with open(memmap_dir / "sample_ids.txt", "rt") as f:
        cached_sample_ids = [line.strip() for line in f]
    for sample_idx, sample in enumerate(dataset.samples):
        expected_id = cached_sample_ids[start_idx + sample_idx]
        assert expected_id == sample.sample_id, "Sample ID validation failed! Offset index = {}, cached sample ID = {}, sliced sample ID = {}".format(
            sample_idx, expected_id, sample.sample_id
        )

    print("Fetching large preallocated tensordict.")
    big_tdict = fetch_preallocated_tdict(memmap_dir)

    print("Populating section")
    populate_memmap_tdict(
        dataset=dataset,
        tdict_start_idx=start_idx,
        big_tdict=big_tdict
    )
