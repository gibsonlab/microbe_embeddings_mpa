import argparse
from typing import *
from pathlib import Path

from skbio.stats.ordination import pcoa
import numpy as np
from numba import njit, prange
import pandas as pd
from gem.datasets.abundance_profile import MetaphlanProfileParser



def main(
        profile_tsv_path: Path,
        metadata_tsv_path: Path,
        out_dir: Path,
        how: str,
        rng_seed: int,
):
    profiles = pd.read_csv(profile_tsv_path, sep="\t")
    profiles_indexed = profiles.set_index("clade_name").transpose()
    profiles_indexed.index.name = "SampleID"
    metadata = pd.read_csv(metadata_tsv_path, sep="\t")

    print("Number of samples (All): {}".format(metadata.shape[0]))
    metadata_subset = metadata.loc[
        (metadata['age_category'] == 'adult')
        & (metadata['disease'] == 'healthy')
    ]
    print("Number of samples (Adult & Healthy): {}".format(metadata_subset.shape[0]))

    # extract the number of SGBs.
    profile_subset_df = select_profiles_in_metadata(metadata_subset, profiles_indexed)
    extractor = MetaphlanProfileParser(profile_subset_df)
    abund_table = extractor.sgb_profile_df.to_numpy()
    num_sgbs_per_sample = np.sum(abund_table > 0, axis=-1)
    assert len(num_sgbs_per_sample.shape) == 1
    assert num_sgbs_per_sample.shape[0] == metadata_subset.shape[0], "The number of samples in the profile DF and metadata DF don't match!"
    num_sgbs_dict = {
        sample_id: n_sgb
        for sample_id, n_sgb in zip(extractor.sgb_profile_df.index, num_sgbs_per_sample)
    }

    sgb_lb = 50
    print("NumSGB statistic: median={}, 0.05={}, 0.95={}".format(
        np.median(num_sgbs_per_sample),
        np.quantile(num_sgbs_per_sample, 0.05),
        np.quantile(num_sgbs_per_sample, 0.95),
    ))
    metadata_subset = metadata_subset.assign(NumSGB=metadata['Sample ID'].map(num_sgbs_dict))
    metadata_subset = metadata_subset.loc[metadata_subset['NumSGB'] >= sgb_lb]
    print("Number of samples with SGB >= {}: {}".format(sgb_lb, metadata_subset.shape[0]))

    # if edge_weight_strategy == "jaccard":
    #     similarity = JaccardSimilarityOracle()
    # elif edge_weight_strategy == "phylogenetic":
    #     similarity = PhylogeneticSimilarityOracle(optional_newick_tree_path)
    # else:
    #     raise ValueError(f"Unrecognized edge_weight_strategy option `{edge_weight_strategy}")
    # train_df, test_df = test_train_split_asv_separation(profiles_indexed, metadata_subset, similarity)
    pcoa_plot_path = out_dir / "pcoa_plot.png"
    if how == "pcoa":
        print("Performing PCoA coordinate-based splitting.")
        train_df, test_df = test_train_split_pcoa_jensenshannon(
            profiles_indexed, metadata_subset,
            train_fraction=0.8,
            test_fraction=0.2,
            plot_path=pcoa_plot_path, train_is_left=False
        )
    elif how == "random":
        print("Performing random splitting.")
        train_df, test_df = test_train_split_random(
            profiles_indexed, metadata_subset,
            train_fraction=0.8,
            test_fraction=0.2,
            rng_seed=rng_seed,
        )

    train_df.to_csv(out_dir / "train.tsv", sep="\t", index=True)
    test_df.to_csv(out_dir / "test.tsv", sep="\t", index=True)
    pd.concat([train_df, test_df], axis=0).to_csv(out_dir / "both.tsv", sep="\t", index=True)

    print("# train samples: {}".format(train_df.shape[0]))
    print("# test samples: {}".format(test_df.shape[0]))
    print("Ratio: {} / {} = {}".format(
        train_df.shape[0], test_df.shape[0], train_df.shape[0] / test_df.shape[0]
    ))


def test_train_split_random(
        profiles_indexed: pd.DataFrame,
        metadata_subset_df: pd.DataFrame,
        rng_seed: int,
        train_fraction: float = 0.8,
        test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profile_df = select_profiles_in_metadata(metadata_subset_df, profiles_indexed)
    rng = np.random.default_rng(rng_seed)
    indices = np.arange(metadata_subset_df.shape[0])
    rng.shuffle(indices)

    # split the indices.
    n_train_rows = int(metadata_subset_df.shape[0] * train_fraction)
    n_test_rows = int(metadata_subset_df.shape[0] * test_fraction)
    train_indices = indices[:n_train_rows]
    test_indices = indices[n_train_rows:n_train_rows + n_test_rows]

    train_metadata_rows = metadata_subset_df.iloc[train_indices]
    test_metadata_rows = metadata_subset_df.iloc[test_indices]

    training_sample_ids = set(train_metadata_rows['Sample ID'])
    test_sample_ids = set(test_metadata_rows['Sample ID'])

    # gather the rows.
    train_df = profile_df[profile_df.index.isin(training_sample_ids)]
    test_df = profile_df[profile_df.index.isin(test_sample_ids)]
    return train_df, test_df


# ================================================ HELPER CODE: misc.
def jaccard_similarity(x: Set, y: Set) -> float:
    numer = len(x.intersection(y))
    denom = len(x.union(y))
    return numer / denom


# def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
#     return scipy_kl_div(p, q).sum()
#
#
# def jensen_shannon_symmetrized_kl(p: np.ndarray, q: np.ndarray) -> float:
#     m = 0.5 * (p + q)
#     return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def select_profiles_in_metadata(metadata_df, profiles_df) -> pd.DataFrame:
    return profiles_df.loc[profiles_df.index.isin(metadata_df['Sample ID'])]


# ================================================= HELPER CODE: pcoa-based splitting.
def test_train_split_pcoa_jensenshannon(
        profiles_indexed: pd.DataFrame,
        metadata_subset_df: pd.DataFrame,
        train_fraction: float = 0.8,
        test_fraction: float = 0.2,
        train_is_left: bool = True,
        plot_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profile_df = select_profiles_in_metadata(metadata_subset_df, profiles_indexed)
    extractor = MetaphlanProfileParser(profile_df)

    abundances = extractor.sgb_profile_df.to_numpy()
    abundances = abundances / abundances.sum(axis=-1, keepdims=True)
    sample_ids = [str(sid) for sid in extractor.sgb_profile_df.index]
    print("Sample IDS preview: {} ...".format(sample_ids[:5]))
    print("# sample IDs = {}".format(len(sample_ids)))

    print("Calculating jensen-shannon distance matrix.")
    dist_mat = calculate_js_distances_numba(abundances)

    """ Compute PCoA. """
    from skbio import DistanceMatrix
    dist_mat = DistanceMatrix(data=dist_mat, ids=sample_ids)
    pcoa_result = pcoa(dist_mat, method='eigh', dimensions=2)
    # noinspection PyTypeHints
    coordinates = pcoa_result.samples.loc[:, ['PC1', 'PC2']].assign(SampleId=sample_ids)
    pc1 = coordinates['PC1'].to_numpy()

    def split_partition_by_pc1(left_q, right_q) -> Tuple[Set[str], Set[str]]:
        # Assign left subset and right subset using the input parameters.
        left_ub = np.quantile(pc1, q=left_q)
        right_lb = np.quantile(pc1, q=right_q)
        partition_left = {sample_id for i, sample_id in enumerate(sample_ids) if pc1[i] < left_ub}
        partition_right = {sample_id for i, sample_id in enumerate(sample_ids) if pc1[i] >= right_lb}
        return partition_left, partition_right


    if train_is_left:
        print("Using tail cutoff quantiles train < {}, test >= {}".format(train_fraction, 1 - test_fraction))
        training_sample_ids, test_sample_ids = split_partition_by_pc1(train_fraction, 1 - test_fraction)
    else:
        print("Using tail cutoff quantiles test < {}, train >= {}".format(test_fraction, 1 - train_fraction))
        test_sample_ids, training_sample_ids = split_partition_by_pc1(test_fraction, 1 - train_fraction)

    train_df = profile_df[profile_df.index.isin(training_sample_ids)]
    test_df = profile_df[profile_df.index.isin(test_sample_ids)]

    # ============================ plot output.
    if plot_path is not None:
        import matplotlib.pyplot as plt
        import seaborn as sb
        test_train_labels = []
        for sample_id in sample_ids:
            if sample_id in training_sample_ids:
                test_train_labels.append("Train")
            elif sample_id in test_sample_ids:
                test_train_labels.append("Test")
            else:
                test_train_labels.append("Excluded")

        coords_with_labels: pd.DataFrame = coordinates.assign(Label=test_train_labels, SampleId=sample_ids)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sb.scatterplot(
            coords_with_labels,
            x='PC1', y='PC2', hue="Label",
            alpha=0.3, linewidth=0., ax=ax
        )

        # Get proportion of variance explained
        prop_var = pcoa_result.proportion_explained
        explained_pc1 = prop_var['PC1']
        explained_pc2 = prop_var['PC2']
        ax.set_xlabel(f'PC1 ({explained_pc1 * 100:.2f}%)')
        ax.set_ylabel(f'PC2 ({explained_pc2 * 100:.2f}%)')
        ax.set_title('PCoA Plot')
        ax.grid(True, alpha=0.3)
        plt.savefig(plot_path, bbox_inches='tight')
        coords_with_labels.to_csv(plot_path.parent / "pcoa_coords.tsv", sep='\t', index=False)
        prop_var.to_csv(plot_path.parent / "pcoa_proportion_explained.tsv", sep='\t', index=True)
    return train_df, test_df


@njit
def kl_divergence_numba(p: np.ndarray, q: np.ndarray) -> float:
    """Numba-optimized KL divergence"""
    result = 0.0
    for i in range(len(p)):
        if p[i] > 0 and q[i] > 0:
            result += p[i] * np.log(p[i] / q[i])  # this is safe, since "q = 0.5*(x+y)" will never be zero in this invocation.
    return result


@njit
def jensen_shannon_symmetrized_kl_numba(p: np.ndarray, q: np.ndarray) -> float:
    """Numba-optimized Jensen-Shannon divergence"""
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence_numba(p, m) + kl_divergence_numba(q, m))


@njit(parallel=True)
def calculate_js_distances_numba(samples: np.ndarray) -> np.ndarray:
    n = samples.shape[0]
    distmat = np.zeros((n, n), dtype=np.float64)

    for i in prange(n):
        for j in range(i + 1, n):
            d = jensen_shannon_symmetrized_kl_numba(samples[i], samples[j])
            distmat[i, j] = d
            distmat[j, i] = d

    return distmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-ft", "--full-table", dest="full_table", type=str, required=True)
    parser.add_argument("-mt", "--metadata-table", dest="metadata_table", type=str, required=True)
    parser.add_argument("-o", "--out-dir", dest="out_dir", type=str, required=True)
    parser.add_argument("-m", "--method", type=str, required=True, help="Either 'pcoa' or 'random'.")
    parser.add_argument("-r", "--rng-seed", dest="rng_seed", type=int, required=False, default=1234, help="Required if using random splitting.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    full_profile_tsv = Path(args.full_table)
    metadata_tsv = Path(args.metadata_table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    main(
        profile_tsv_path=full_profile_tsv,
        metadata_tsv_path=metadata_tsv,
        out_dir=out_dir,
        how=args.method,
        rng_seed=args.rng_seed,
    )
