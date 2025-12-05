from pathlib import Path
import zstandard as zstd
import pandas as pd
import torch

from gem.mpa.embeddings import MetaphlanMarkerEmbedding
from gem.mpa.dataset import MetaphlanDataset


def generate_test_profile() -> pd.DataFrame:
    with zstd.open("example.tsv.zst", "rt") as f:
        return pd.read_csv(f, sep='\t')


def test_dataset():
    test_embed = MetaphlanMarkerEmbedding(
        marker_embedding_memmap_dir=Path("/data/local/youn/metaphlan_dset/embeddings/phylophlan_markers/evo/memmap"),
    )
    test_dset = MetaphlanDataset(
        generate_test_profile(),
        test_embed,
        max_num_sgbs=100
    )
    assert len(test_dset) == 1
    sgbs, f, m, s, t = test_dset.load_sample_embeddings(test_dset.samples[0])
    assert f.shape[0] == test_dset.max_num_sgbs
    assert f.shape[1] == test_dset.max_num_markers
    assert f.shape[2] == 4096
    assert m.shape[0] == test_dset.max_num_sgbs
    assert m.shape[1] == test_dset.max_num_markers
    assert s.shape[0] == test_dset.max_num_sgbs
    assert t.shape[0] == test_dset.max_num_sgbs

    # ensure that the number of non-padded SGBs equals the number of nonzero feature vectors.
    assert torch.sum(t != 0.0) == len(sgbs)


if __name__ == "__main__":
    test_dataset()
