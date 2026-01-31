from pathlib import Path
from gem.datasets.mpa.embeddings import MetaphlanMarkerEmbedding


def test_embeddings():
    test_embed = MetaphlanMarkerEmbedding(
        marker_embedding_memmap_dir=Path("/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo/memmap"),
    )

    f, p = test_embed.convert_sgb("SGB10130", max_markers=500)  # should be ok, because it has no markers in database.
    f, p = test_embed.convert_sgb("SGB124", max_markers=500)
    assert (f.shape[0], f.shape[1]) == (500, 4096)
    assert p.shape[0] == 500


if __name__ == "__main__":
    test_embeddings()
