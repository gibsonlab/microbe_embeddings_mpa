from pathlib import Path
import zstandard as zstd
import pandas as pd
import torch

from gem.mpa.embeddings import MetaphlanMarkerEmbedding
from gem.mpa.dataset import MetaphlanDataset
from gem.ml.dataloader.collate import BufferedCollator


def generate_test_profile() -> pd.DataFrame:
    with zstd.open("example.tsv.zst", "rt") as f:
        return pd.read_csv(f, sep='\t')


def test_dataset():
    test_embed = MetaphlanMarkerEmbedding(
        marker_embedding_basedir=Path("/data/local/youn/metaphlan_abundance_prediction/embedding/evo"),
    )
    test_dset = MetaphlanDataset(
        generate_test_profile(),
        test_embed,
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
    assert len(sgbs) == torch.sum(t != 0.0)
    assert len(sgbs) == torch.sum(s)
    assert len(sgbs) == (f == 0).all(dim=1).sum()

def test_collator():
    test_embed = MetaphlanMarkerEmbedding(
        marker_embedding_basedir=Path("/data/local/youn/metaphlan_abundance_prediction/embedding/evo"),
    )
    test_dset = MetaphlanDataset(
        generate_test_profile(),
        test_embed,
    )
    collator = BufferedCollator(
        batch_size=1,
        max_markers=test_dset.max_num_markers,
        max_num_sgbs=test_dset.max_num_sgbs,
        embed_feature_dim=test_dset.embed_feature_dim,
        dtype=test_dset.embedding_dtype
    )

    sgbs, f, m, s, t = test_dset.load_sample_embeddings(test_dset.samples[0])
    f_col, m_col, s_col, t_col = collator(batch=[test_dset[0]])
    assert f_col.shape[1] == test_dset.max_num_sgbs
    assert f.shape[2] == test_dset.max_num_markers
    assert f.shape[3] == 4096
    assert m.shape[1] == test_dset.max_num_sgbs
    assert m.shape[2] == test_dset.max_num_markers
    assert s.shape[1] == test_dset.max_num_sgbs
    assert t.shape[1] == test_dset.max_num_sgbs


if __name__ == "__main__":
    test_dataset()
