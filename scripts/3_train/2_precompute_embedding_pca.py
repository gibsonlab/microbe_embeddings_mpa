from pathlib import Path
from gem.mpa import MetaphlanMarkerEmbedding


def main(target_dim: int, ipca_batch_size: int):
    embed_basedir = Path("/data/local/youn/metaphlan_abundance_prediction/embedding/evo")
    embedding = MetaphlanMarkerEmbedding(
        embed_basedir,
        dimension_reduce_pca=target_dim,
        ipca_batch_size=ipca_batch_size
    )


if __name__ == "__main__":
    main(
        target_dim=768,
        ipca_batch_size=10000
    )
