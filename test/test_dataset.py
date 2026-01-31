from pathlib import Path
import zstandard as zstd
import pandas as pd
import torch

from gem.ml import MetaphlanDataLoader
from gem.datasets.mpa.embeddings import MetaphlanMarkerEmbedding
from gem.datasets.mpa.dataset import MetaphlanDataset
from gem.ml.dataloader.collate import BufferedCollator


def generate_test_profile() -> pd.DataFrame:
    with zstd.open("example.tsv.zst", "rt") as f:
        df = pd.read_csv(f, sep='\t', index_col='SampleID')
        df['t__SGBFAKETEST'] = 100.0
        return df


def test_dataset(marker_embedding_basedir: Path):
    test_embed = MetaphlanMarkerEmbedding(marker_embedding_basedir=marker_embedding_basedir)
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

        
def test_collator(marker_embedding_basedir: Path):
    test_embed = MetaphlanMarkerEmbedding(marker_embedding_basedir=marker_embedding_basedir)
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

    sample_ids_col, f_col, m_col, s_col, t_col = collator(batch=test_batch)

    # expected answers
    s_dim = max([len(x.sgb_ids) for x in test_dset.samples])
    m_dim = max([
        test_embed.num_markers(sgb)
        for sample in test_dset.samples
        for sgb in sample.sgb_ids
        if test_embed.contains_sgb(sgb)
    ])
    e_dim = 4096
    assert f_col.shape == (len(test_batch), s_dim, m_dim, e_dim)
    assert m_col.shape == (len(test_batch), s_dim, m_dim)
    assert s_col.shape == (len(test_batch), s_dim)
    assert t_col.shape == (len(test_batch), s_dim)
    
    for i in range(len(test_dset.samples)):
        sample = test_dset.samples[i]
        sgbs = sample.sgb_ids
        _, f, m, s, t = test_batch[i]

        assert f_col[i, :f.shape[0], :f.shape[1], :f.shape[2]].equal(f)
        n_fake = 1
        n_true = len(sgbs) - n_fake
        assert n_true == torch.sum(s_col[i]).item()
        assert n_true == m_col[i].any(dim=-1).sum().item()
        for s_mask, s_feats in zip(s_col[i], f_col[i]):
            if s_mask:
                assert (s_feats != 0).any().item()
        assert torch.isclose(torch.sum(s_col[i] * t_col[i]), torch.tensor(1.0)), "Expected abundances to approx. sum to 1. Sample = {}, got: {}".format(
            i, torch.sum(s_col[i] * t_col[i])
        )


def test_dataloader(marker_embedding_basedir: Path):
    test_embed = MetaphlanMarkerEmbedding(marker_embedding_basedir=marker_embedding_basedir)
    test_dset = MetaphlanDataset(
        generate_test_profile(),
        test_embed,
    )

    batch_sz = 1
    n_workers = 2
    test_dloader = MetaphlanDataLoader(
        dataset=test_dset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        worker_rng_seed=314159
    )

    samples_by_name = {sample.sample_id: sample for sample in test_dset.samples}

    e_dim = 4096
    for (sample_ids, f_batch, m_batch, s_batch, t_batch) in test_dloader:
        batch_samples = [samples_by_name[sid] for sid in sample_ids]
        s_dim = max(len(sample.sgb_ids) for sample in batch_samples)
        m_dim = max(
            test_embed.num_markers(sgb_id)
            for sample in batch_samples
            for sgb_id in sample.sgb_ids
            if test_embed.contains_sgb(sgb_id)
        )
        assert f_batch.shape == (batch_sz, s_dim, m_dim, e_dim)
    

if __name__ == "__main__":
    # marker_embedding_dir = Path("/data/local/youn/metaphlan_abundance_prediction/embedding/evo")
    marker_embedding_dir = Path("/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo")
    #test_dataset(marker_embedding_dir)
    test_collator(marker_embedding_dir)
    test_dataloader(marker_embedding_dir)
