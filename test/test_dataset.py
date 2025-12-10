from pathlib import Path
import zstandard as zstd
import pandas as pd
import torch

from gem.mpa.embeddings import MetaphlanMarkerEmbedding
from gem.mpa.dataset import MetaphlanDataset
from gem.ml.dataloader.collate import BufferedCollator


def generate_test_profile() -> pd.DataFrame:
    with zstd.open("example.tsv.zst", "rt") as f:
        df = pd.read_csv(f, sep='\t', index_col='SampleID')
        df['t__SGBFAKETEST'] = 100.0
        return df


def test_dataset():
    test_embed = MetaphlanMarkerEmbedding(
        marker_embedding_basedir=Path("/data/local/youn/metaphlan_abundance_prediction/embedding/evo"),
    )
    test_dset = MetaphlanDataset(
        generate_test_profile(),
        test_embed,
    )
    assert len(test_dset) == 2

    for i in range(len(test_dset.samples)):
        sample = test_dset.samples[i]
        sgbs, f, m, s, t = test_dset.load_sample_embeddings(sample)

        # expected answers
        s_dim = len(sample.sgb_ids)        
        m_dim = max(test_embed.num_markers(s) for s in sample.sgb_ids if test_embed.contains_sgb(s))
        e_dim = 4096
        
        assert f.shape == (s_dim, m_dim, e_dim)
        assert m.shape == (s_dim, m_dim)
        assert s.shape == (s_dim,)
        assert t.shape == (s_dim,)

        # ensure that the number of non-padded SGBs equals the number of nonzero feature vectors.
        n_fake = 1
        n_true = len(sample.sgb_ids) - n_fake
        assert n_true == torch.sum(s).item()
        assert n_true == m.any(dim=-1).sum().item()
        assert n_fake == (f == 0).all(dim=[1, 2]).sum().item()

        
def test_collator():
    test_embed = MetaphlanMarkerEmbedding(
        marker_embedding_basedir=Path("/data/local/youn/metaphlan_abundance_prediction/embedding/evo"),
    )
    test_dset = MetaphlanDataset(
        generate_test_profile(),
        test_embed,
    )

    test_batch = [test_dset[0], test_dset[1]]
    collator = BufferedCollator(
        batch_size=len(test_batch),
        max_markers=test_dset.max_num_markers(),
        max_num_sgbs=test_dset.max_num_sgbs(),
        embed_feature_dim=test_dset.embed_feature_dim(),
        dtype=test_dset.embedding_dtype()
    )

    f_col, m_col, s_col, t_col = collator(batch=test_batch)
    # expected answers
    s_dim = test_dset.max_num_sgbs()
    m_dim = test_dset.max_num_markers()
    e_dim = 4096
    assert f_col.shape == (len(test_batch), s_dim, m_dim, e_dim)
    assert m_col.shape == (len(test_batch), s_dim, m_dim)
    assert s_col.shape == (len(test_batch), s_dim)
    assert t_col.shape == (len(test_batch), s_dim)
    
    for i in range(len(test_dset.samples)):
        sample = test_dset.samples[i]
        sgbs = sample.sgb_ids
        f, m, s, t = test_batch[i]

        assert f_col[i, :f.shape[0], :f.shape[1], :f.shape[2]].equal(f)
        n_fake = 1
        n_true = len(sample.sgb_ids) - n_fake
        assert n_true == torch.sum(s_col[i]).item()
        assert n_true == m_col[i].any(dim=-1).sum().item()
        for s_mask, s_feats in zip(s_col[i], f_col[i]):
            if s_mask:
                assert (s_feats != 0).any().item()
        assert torch.isclose(torch.sum(s_col[i] * t_col[i]), torch.tensor(1.0)), "Expected abundances to approx. sum to 1. Sample = {}, got: {}".format(
            i, torch.sum(s_col[i] * t_col[i])
        )
    

if __name__ == "__main__":
    test_dataset()
    test_collator()
