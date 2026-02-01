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
from gem.datasets import MetaphlanMarkerPrecomputedEmbedding, perform_allocation, MetaphlanPreembeddedDataset, MetaphlanProfileParser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-tsv", dest="dataset_tsv", required=True, type=str)
    parser.add_argument("-e", "--embedding-dir", dest="marker_embedding_basedir", required=True, type=str)
    parser.add_argument("-m", "--memmap-dir", dest="memmap_dir", required=True, type=str)
    parser.add_argument("-t", "--threads", dest="num_threads", required=False, type=int, default=1)
    parser.add_argument("--start", dest="start_row", required=False, type=int, default=0)  # inclusive
    parser.add_argument("--end", dest="end_row", required=False, type=int, default=-1)  # inclusive
    parser.add_argument("--padding", dest="add_padding", required=False, action="store_true", default=False)
    parser.add_argument("--dimension-reduce", dest="dimension_reduce_pca", required=False, default=None, type=int,
                        help="If specified (an integer greater than zero), will perform incremental PCA on the entire"
                             "set of embeddings for dimensionality reduction.")
    parser.add_argument("--pca-batch-size", dest="ipca_batch_size", required=False, default=10000, type=int,
                        help="Specify the batch size for incremental PCA. Default: 10000")
    return parser.parse_args()


def main(
        dataset_df: pd.DataFrame,
        marker_embedding: MetaphlanMarkerPrecomputedEmbedding,
        memmap_dir: Path,
        add_padding: bool,
        max_num_markers: int,
        num_workers: int,
):
    memmap_dir.mkdir(parents=True, exist_ok=True)
    regular_dset = MetaphlanPreembeddedDataset(dataset_df, marker_embedding)
    if add_padding:
        print(f"[***] NOTE ---> Marker-dim will be padded into: {max_num_markers}")
    perform_allocation(
        dataset=regular_dset,
        cache_dir=memmap_dir,
        num_threads=num_workers,
        add_padding=add_padding,
        max_num_markers=max_num_markers,
    )
    print("Finished memory-mapping tensors.")


if __name__ == "__main__":
    args = parse_args()
    dataset_df_full = pd.read_csv(args.dataset_tsv, sep='\t', index_col="SampleID")
    start_idx = args.start_row - 1
    if args.end_row == -1:
        print(f"Executing allocation for row #{args.start_row} onwards.")
        dataset_df = dataset_df_full.iloc[start_idx:]
    else:
        print(f"Executing allocation for row #{args.start_row} ~ #{args.end_row} (inclusive).")
        end_idx = args.end_row
        dataset_df = dataset_df_full.iloc[start_idx:end_idx]

    marker_embedding = MetaphlanMarkerPrecomputedEmbedding(
        marker_embedding_basedir=Path(args.marker_embedding_basedir),
        dimension_reduce_pca=args.dimension_reduce_pca,
        ipca_batch_size=args.ipca_batch_size,
    )

    all_samples = list(MetaphlanProfileParser(dataset_df_full).samples())
    max_num_markers = max(
        marker_embedding.num_markers(sgb_id)
        for sample in all_samples
        for sgb_id in sample.sgb_ids
        if marker_embedding.contains_sgb(sgb_id)
    )

    main(
        dataset_df=dataset_df,
        marker_embedding=marker_embedding,
        memmap_dir=Path(args.memmap_dir),
        num_workers=args.num_threads,
        add_padding=args.add_padding,
        max_num_markers=max_num_markers,
    )