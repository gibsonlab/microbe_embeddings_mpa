from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import trange


def dim_reduce_ipca(
        input_tensor_path: Path,
        out_tensor_path: Path,
        target_dim: int,
        batch_size: int = 10_000,
):
    feature_tensor = torch.load(input_tensor_path)
    print(f"Got input tensor of shape: {feature_tensor.shape}")

    transformed_tensor = incremental_pca_on_tensor(feature_tensor, target_dim, batch_size)
    print(f"Got transformed tensor of shape: {transformed_tensor.shape}")

    torch.save(transformed_tensor, out_tensor_path)
    print(f"Wrote new tensor to: {out_tensor_path}")


def incremental_pca_on_tensor(
        feature_tensor: torch.Tensor,
        target_dim: int,
        batch_size: int,
) -> torch.Tensor:
    """
    Perform Incremental PCA on a (*, feature_dim) tensor.
    Returns a reduced tensor of shape (*, target_dim).
    """
    if target_dim > feature_tensor.shape[-1]:
        raise ValueError("Can't dim-reduce a tensor of feature dim {} into {}, which is larger.".format(
            feature_tensor.shape[-1], target_dim
        ))

    original_shape = feature_tensor.shape
    feature_dim = original_shape[-1]

    # Flatten to 2D: (1_743_000, 768)
    X_torch = feature_tensor.reshape(-1, feature_dim)
    X_all = X_torch.float().detach().cpu().numpy()  # Note: this converts "bfloat16" to "float32" if necessary.
    print("Total vectors: {}".format(X_all.shape[0]))

    X_valid = X_all[~np.isnan(X_all).any(axis=-1), :]
    print("Non-NaN vectors: {}".format(X_valid.shape[0]))

    n_total_to_embed = X_valid.shape[0]

    # Fit IncrementalPCA
    ipca = IncrementalPCA(n_components=target_dim, batch_size=batch_size)

    print("Fitting Incremental-PCA...")
    for start in trange(0, n_total_to_embed, batch_size, desc='iPCA:Fit'):
        batch = X_valid[start : start + batch_size]
        ipca.partial_fit(batch)

    # Transform in batches (avoids materializing full output at once)
    print("Transforming...")
    chunks = []
    for start in trange(0, n_total_to_embed, batch_size, desc='iPCA:Transform'):
        batch = X_all[start : start + batch_size]
        valid_idxs, = np.where(~np.isnan(batch).any(axis=-1))

        batch_all_xformed = np.full(batch.shape, fill_value=np.nan, dtype=batch.dtype)
        if len(valid_idxs) > 0:
            batch_all_xformed[valid_idxs, :] = ipca.transform(batch[valid_idxs, :])

        chunks.append(batch_all_xformed)

    X_reduced = np.concatenate(chunks, axis=0)  # (1_743_000, n_components)

    # Convert back to torch and reshape
    result = torch.from_numpy(X_reduced).reshape(*original_shape[:-1], target_dim)   # shape: (*, target_dim)
    print(f"Done. Explained variance retained: {ipca.explained_variance_ratio_.sum():.3%}")
    return result


if __name__ == "__main__":
    embed_dir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings")
    reduced_dim = 100
    dim_reduce_ipca(embed_dir / "phylophlan" /  "dnabert-s.pt",
                    embed_dir / "phylophlan" / f"dnabert-s.ipca_d{reduced_dim}.pt", target_dim=reduced_dim)
    dim_reduce_ipca(embed_dir / "phylophlan" / "evo-1-8k-base_hyena5.pt",
                    embed_dir / "phylophlan" / f"evo-1-8k-base_hyena5.ipca_d{reduced_dim}.pt", target_dim=reduced_dim)
    dim_reduce_ipca(embed_dir / "phylophlan" / "evo2_7b_hyena10.pt",
                    embed_dir / "phylophlan" / f"evo2_7b_hyena10.ipca_d{reduced_dim}.pt", target_dim=reduced_dim)

    dim_reduce_ipca(embed_dir / "phylophlan_metaphlan" / "dnabert-s.pt",
                    embed_dir / "phylophlan_metaphlan" / f"dnabert-s.ipca_d{reduced_dim}.pt", target_dim=reduced_dim)
    dim_reduce_ipca(embed_dir / "phylophlan_metaphlan" / "evo-1-8k-base_hyena5.pt",
                    embed_dir / "phylophlan_metaphlan" / f"evo-1-8k-base_hyena5.ipca_d{reduced_dim}.pt", target_dim=reduced_dim)
    dim_reduce_ipca(embed_dir / "phylophlan_metaphlan" / "evo2_7b_hyena10.pt",
                    embed_dir / "phylophlan_metaphlan" / f"evo2_7b_hyena10.ipca_d{reduced_dim}.pt", target_dim=reduced_dim)

