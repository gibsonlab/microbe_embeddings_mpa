import argparse
from pathlib import Path

import pandas as pd
import torch
from gem.datasets.mpa.dataset_memmap_large import allocate_big_memmap_tdict

from gem.datasets.mpa import MetaphlanProfileCollection, MetaphlanMarkerEmbedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-tsv", dest="dataset_tsv", required=True, type=str)
    parser.add_argument("-e", "--embedding-dir", dest="marker_embedding_basedir", required=True, type=str)
    parser.add_argument("-m", "--memmap-dir", dest="memmap_dir", required=True, type=str)
    parser.add_argument("--dimension-reduce", dest="dimension_reduce_pca", required=False, default=None, type=int,
                        help="If specified (an integer greater than zero), will perform incremental PCA on the entire"
                             "set of embeddings for dimensionality reduction.")
    parser.add_argument("--pca-batch-size", dest="ipca_batch_size", required=False, default=10000, type=int,
                        help="Specify the batch size for incremental PCA. Default: 10000")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_df_full = pd.read_csv(args.dataset_tsv, sep='\t', index_col="SampleID")
    all_samples = list(MetaphlanProfileCollection(dataset_df_full).samples())

    marker_embedding = MetaphlanMarkerEmbedding(
        marker_embedding_basedir=Path(args.marker_embedding_basedir),
        dimension_reduce_pca=args.dimension_reduce_pca,
        ipca_batch_size=args.ipca_batch_size,
    )

    max_num_sgbs = max(
        len(sample.sgb_ids)
        for sample in all_samples
    )

    max_num_markers = max(
        marker_embedding.num_markers(sgb_id)
        for sample in all_samples
        for sgb_id in sample.sgb_ids
        if marker_embedding.contains_sgb(sgb_id)
    )

    allocate_big_memmap_tdict(
        sample_ids=[s.sample_id for s in all_samples],
        out_dir=Path(args.memmap_dir),
        S_max_global=max_num_sgbs,
        M_max_global=max_num_markers,
        embed_dim=marker_embedding.embedding_dim,
        dtype=torch.float32,
    )
