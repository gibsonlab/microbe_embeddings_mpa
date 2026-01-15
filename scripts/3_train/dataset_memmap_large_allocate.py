import argparse
from pathlib import Path
from typing import *
import json

import pandas as pd
import torch
from tensordict import TensorDict, MemoryMappedTensor

from gem.mpa import MetaphlanProfileExtractor, MetaphlanMarkerEmbedding


def allocate_big_memmap_tdict(
    sample_ids: List[str],
    out_dir: Path,
    S_max_global: int,
    M_max_global: int,
    embed_dim: int,
    dtype: torch.dtype = torch.float32,
):
    """
    Allocates a big memmapped tensordict, with the requested shape and dtype.

    :param dataset:
    :param out_dir:
    :param S_max_global:
    :param M_max_global:
    :param embed_dim:
    :param f_dtype:
    :param t_dtype:
    :return:
    """
    if dtype == torch.float32:
        dtype_str = "torch.float32"
    else:
        raise ValueError(f"This script currently does not support the dtype {dtype}")

    print(f"Target allocation path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(sample_ids)

    # 1) Allocate on-disk MemoryMappedTensors
    print("Feature shape: ({}, {}, {}, {})".format(N, S_max_global, M_max_global, embed_dim))
    big_features = MemoryMappedTensor.empty(
        (N, S_max_global, M_max_global, embed_dim),
        dtype=dtype,
        filename=str(out_dir / "features.mmap"),
    )
    big_mpadding = MemoryMappedTensor.empty(
        (N, S_max_global, M_max_global),
        dtype=torch.bool,
        filename=str(out_dir / "mpadding.mmap"),
    )
    big_spadding = MemoryMappedTensor.empty(
        (N, S_max_global),
        dtype=torch.bool,
        filename=str(out_dir / "spadding.mmap"),
    )
    big_targets = MemoryMappedTensor.empty(
        (N, S_max_global),
        dtype=dtype,
        filename=str(out_dir / "targets.mmap"),
    )

    big_td = TensorDict(
        {
            "features": big_features,
            "mpadding": big_mpadding,
            "spadding": big_spadding,
            "targets": big_targets,
        },
        batch_size=[N],
        device="cpu",
    )

    with open(out_dir / "sample_ids.txt", "wt") as f:
        for s_id in sample_ids:
            print(s_id, file=f)

    with open(out_dir / "meta.json", "wt") as f:
        json.dump(
            {
                "N": N,
                "S": S_max_global,
                "M": M_max_global,
                "E": embed_dim,
                "dtype": dtype_str
            },
            f
        )

    return big_td


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
    all_samples = list(MetaphlanProfileExtractor(dataset_df_full).samples())

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
