"""
Script for pre-computing evo embeddings for marker genes.
"""
from typing import *
import logging
import sys
import argparse
from pathlib import Path

import torch
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", dest="method", type=str)
    parser.add_argument("-f", "--full-sgbs", dest="full_sgb_list", type=str)
    parser.add_argument("-e", "--embed-dim", dest="embed_dim", type=int)
    parser.add_argument("-s", "--seed", dest="rng_seed", type=int)
    parser.add_argument("-d", "--distance-matrix", dest="distance_matrix", type=str)
    parser.add_argument("-o", "--out", dest="out_path", type=str, help='A path to the h5 file to create.')
    return parser.parse_args()


def embed_umap(
        rng_seed: int,
        embed_dim: int,
        distance_matrix: np.ndarray,
) -> np.ndarray:
    from umap import UMAP
    umap_model = UMAP(
        random_state=rng_seed,
        n_components=embed_dim,
        metric='precomputed',
    )
    embeddings = umap_model.fit_transform(distance_matrix)
    return embeddings


def embed_pcoa(
        rng_seed: int,
        embed_dim: int,
        distance_matrix: np.ndarray,
        sgb_id_order: List[str],
) -> np.ndarray:
    from skbio import DistanceMatrix
    from skbio.stats.ordination import pcoa
    distance_matrix = DistanceMatrix(
        data=distance_matrix,
        ids=sgb_id_order,
    )
    pcoa_results = pcoa(distance_matrix, dimensions=embed_dim, seed=rng_seed)
    coordinates = pcoa_results.samples.values
    return coordinates


def store_embeddings(
        full_sgb_ids: List[str],
        embeddings: np.ndarray,
        embedding_id_order: List[str],
        output_path: Path
):
    assert len(embedding_id_order) == embeddings.shape[0]
    embed_dim = embeddings.shape[1]
    if len(full_sgb_ids) != len(embedding_id_order):
        leftover = set(full_sgb_ids).difference(set(embedding_id_order))
        logger.warning("Following {} SGBs were not provided in the distance matrix: {}".format(
            len(leftover),
            ",".join(str(s) for s in leftover)
        ))
    else:
        leftover = set()
    logger.info(f"Storing {len(embedding_id_order)} embeddings of dimension {embed_dim} into {output_path}")

    # Initialize empty memmap
    tensor_shape = (len(full_sgb_ids), 1, embed_dim)
    print("Allocating stacked embeddings tensor of shape {}".format(tensor_shape))
    print("Save target: {}".format(output_path))
    full_tensor = torch.full(
        tensor_shape,
        fill_value=torch.nan,
        dtype=torch.float32,
    )

    embedding_order = {s_id: _i for _i, s_id in enumerate(embedding_id_order)}
    for sgb_idx, sgb_id in enumerate(full_sgb_ids):
        if sgb_id in embedding_order:
            _i = embedding_order[sgb_id]
            full_tensor[sgb_idx, 0, :] = torch.from_numpy(embeddings[_i])
        else:
            full_tensor[sgb_idx, 0, :] = torch.nan

    with open(output_path.with_suffix(".meta"), "wt") as meta_f:
        print("float32", file=meta_f)
        print(','.join(str(s) for s in tensor_shape), file=meta_f)
        print("MISSING={}".format(len(leftover)), file=meta_f)
        for s_id in leftover:
            print(s_id, file=meta_f)
    with open(output_path.with_suffix(".sgb.txt"), "wt") as sgb_f:
        for sgb_id in full_sgb_ids:
            print(sgb_id, file=sgb_f)

    torch.save(full_tensor, output_path)
    print("Saved tensor to {}".format(output_path))


def do_job(
        method: str,
        full_id_path: Path,
        embed_dim: int,
        rng_seed: int,
        distance_matrix_file: Path,
        out_path: Path,
):
    with open(full_id_path, "rt") as f:
        full_ids = [l.strip() for l in f if len(l.strip()) > 0]

    logger.info(f"Target SGB embedding output directory: {out_path}")
    dist_mat_zip = np.load(distance_matrix_file)
    dist_mat = dist_mat_zip['mat']
    embedding_order = dist_mat_zip['labels']
    if method == 'umap':
        logger.info(f"Computing UMAP embeddings (d={embed_dim}, seed={rng_seed})")
        embeddings = embed_umap(rng_seed, embed_dim, dist_mat)
    elif method == 'pcoa':
        logger.info(f"Computing PCoA embeddings (d={embed_dim}, seed={rng_seed})")
        embeddings = embed_pcoa(rng_seed, embed_dim, dist_mat, embedding_order)
    else:
        raise ValueError(f"Unrecognized embedding method `{method}`")

    logger.info("Storing results.")
    store_embeddings(full_ids, embeddings, embedding_order, out_path)


if __name__ == "__main__":
    args = parse_args()

    do_job(
        method=args.method,
        full_id_path=Path(args.full_sgb_list),
        embed_dim=args.embed_dim,
        rng_seed=args.rng_seed,
        distance_matrix_file=Path(args.distance_matrix),
        out_path=Path(args.out_path),
    )
