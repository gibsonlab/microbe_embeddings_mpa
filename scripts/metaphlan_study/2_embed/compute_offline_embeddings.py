"""
Script for pre-computing evo embeddings for marker genes.
"""
from typing import *
import logging
import sys
import argparse
from pathlib import Path

import h5py
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
    parser.add_argument("-e", "--embed-dim", dest="embed_dim", type=str)
    parser.add_argument("-s", "--seed", dest="rng_seed", type=str)
    parser.add_argument("-d", "--distance-matrix", dest="distance_matrix", type=Path)
    parser.add_argument("-o", "--out", dest="out_path", type=str)
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


def store_embeddings(embeddings: np.ndarray, sgb_id_order: List[str], out_path: Path):
    assert len(sgb_id_order) == embeddings.shape[0]
    embed_dim = embeddings.shape[1]
    logger.info(f"Storing {len(sgb_id_order)} embeddings of dimension {embed_dim} into {out_path}")

    with h5py.File(out_path, 'w') as f:
        for sgb_id, sgb_embedding in zip(sgb_id_order, embeddings):
            f.create_dataset(sgb_id, data=sgb_embedding, compression='lzf')


def do_job(
        method: str,
        embed_dim: int,
        rng_seed: int,
        distance_matrix_file: Path,
        out_path: Path,
):
    logger.info("Target SGB embedding output directory: ")
    dist_mat_zip = np.load(distance_matrix_file)
    dist_mat = dist_mat_zip['mat']
    sgb_id_order = dist_mat_zip['labels']
    if method == 'umap':
        logger.info(f"Computing UMAP embeddings (d={embed_dim}, seed={rng_seed})")
        embeddings = embed_umap(rng_seed, embed_dim, dist_mat)
    elif method == 'pcoa':
        logger.info(f"Computing PCoA embeddings (d={embed_dim}, seed={rng_seed})")
        embeddings = embed_pcoa(rng_seed, embed_dim, dist_mat, sgb_id_order)
    else:
        raise ValueError(f"Unrecognized embedding method `{method}`")

    logger.info("Storing results.")
    store_embeddings(embeddings, sgb_id_order, out_path)


if __name__ == "__main__":
    args = parse_args()

    do_job(
        method=args.method,
        embed_dim=args.embed_dim,
        rng_seed=args.rng_seed,
        distance_matrix_file=Path(args.distance_matrix),
        out_path=Path(args.out_path),
    )
