import argparse
from pathlib import Path
from gem.datasets.mpa import MetaphlanMarkerEmbedding


def main(embed_basedir: Path, target_dim: int, ipca_batch_size: int):
    _ = MetaphlanMarkerEmbedding(
        embed_basedir,
        dimension_reduce_pca=target_dim,
        ipca_batch_size=ipca_batch_size
    )
    print("Done -- Instantiated and precomputed PCA model.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-embed-dir', dest='embed_basedir', type=str, required=True)
    parser.add_argument("--dimension-reduce", dest="dimension_reduce_pca", required=True, type=int,
                        help="The target output dimension of the PCA on the entire set of embeddings for dimensionality reduction.")
    parser.add_argument("--pca-batch-size", dest="ipca_batch_size", required=False, default=10000, type=int,
                        help="Specify the batch size for incremental PCA. Default: 10000")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        embed_basedir=Path(args.embed_basedir),
        target_dim=args.dimension_reduce_pca,
        ipca_batch_size=args.ipca_batch_size,
    )
