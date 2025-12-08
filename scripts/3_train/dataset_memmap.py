"""
Pre-compute the tensors in the dataset, and convert it into memory-mapped tensordicts.
This is meant to reduce the time it takes to dynamically re-alloate memory for each sample.
"""

"""
From https://docs.pytorch.org/tensordict/main/saving.html

1) Saving a memmapped tensordict:
    x = TensorDict()
    x_disk = x.memmap("/path/to/saved/dir", num_threads=30)

2) Loading a memmapped tensordict:
    x = TensorDict.load_memmap("/path/to/saved/dir")
"""
import argparse
from pathlib import Path

import pandas as pd
from gem.mpa import MetaphlanMarkerEmbedding
from gem.mpa import perform_allocation, MetaphlanDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-tsv", dest="dataset_tsv", required=True, type=str)
    parser.add_argument("-e", "--embedding-dir", dest="marker_embedding_basedir", required=True, type=str)
    parser.add_argument("-m", "--memmap-dir", dest="memmap_dir", required=True, type=str)
    parser.add_argument("-t", "--threads", dest="num_threads", required=False, type=int, default=1)
    parser.add_argument("--start", dest="start_row", required=False, type=int, default=0)  # inclusive
    parser.add_argument("--end", dest="end_row", required=False, type=int, default=-1)  # inclusive
    return parser.parse_args()


def main(
        dataset_df: pd.DataFrame,
        marker_embedding: MetaphlanMarkerEmbedding,
        memmap_dir: Path,
        num_workers: int,
):
    memmap_dir.mkdir(parents=True, exist_ok=True)
    regular_dset = MetaphlanDataset(dataset_df, marker_embedding)
    perform_allocation(
        dataset=regular_dset,
        cache_dir=memmap_dir,
        num_threads=num_workers
    )
    print("Finished memory-mapping tensors.")


if __name__ == "__main__":
    args = parse_args()
    dataset_df = pd.read_csv(args.dataset_tsv, sep='\t', index_col="SampleID")
    start_idx = args.start_row - 1
    if args.end_row == -1:
        print(f"Executing allocation for row #{args.start_row} onwards.")
        dataset_df = dataset_df.iloc[start_idx:]
    else:
        print(f"Executing allocation for row #{args.start_row} ~ #{args.end_row} (inclusive).")
        end_idx = args.end_row
        dataset_df = dataset_df.iloc[start_idx:end_idx]

    main(
        dataset_df=dataset_df,
        marker_embedding=MetaphlanMarkerEmbedding(marker_embedding_basedir=Path(args.marker_embedding_basedir)),
        memmap_dir=Path(args.memmap_dir),
        num_workers=args.num_threads,
    )