import argparse
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from gem.datasets.mpa.dataset_memmap_large import fetch_preallocated_tdict, parse_sample_ids_memmap
from tensordict import TensorDict

from gem.datasets.mpa import MetaphlanPreembeddedDataset, MetaphlanMarkerPrecomputedEmbedding


def populate_memmap_tdict(
        dataset: MetaphlanPreembeddedDataset,
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

    marker_embedding = MetaphlanMarkerPrecomputedEmbedding(
        marker_embedding_basedir=Path(args.marker_embedding_basedir),
        dimension_reduce_pca=args.dimension_reduce_pca,
        ipca_batch_size=args.ipca_batch_size,
    )
    dset = MetaphlanPreembeddedDataset(dataset_df, marker_embedding)

    # validate sample ID ordering.
    print("Validating sample ordering.")
    cached_sample_ids = parse_sample_ids_memmap(memmap_dir)
    for s_idx, _sample in enumerate(dset.samples):
        expected_id = cached_sample_ids[start_idx + s_idx]
        assert expected_id == _sample.sample_id, "Sample ID validation failed! Offset index = {}, cached sample ID = {}, sliced sample ID = {}".format(
            s_idx, expected_id, _sample.sample_id
        )

    print("Fetching large preallocated tensordict.")
    prealloc_tdict, S_max, M_max, E, dtype = fetch_preallocated_tdict(memmap_dir)

    print("Populating section")
    populate_memmap_tdict(
        dataset=dset,
        tdict_start_idx=start_idx,
        big_tdict=prealloc_tdict
    )
