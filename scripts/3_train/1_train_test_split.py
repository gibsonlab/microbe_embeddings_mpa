from abc import abstractmethod, ABC
from typing import *
from pathlib import Path
import itertools
from tqdm import tqdm

from Bio import Phylo
from Bio.Phylo.Newick import Tree

import numpy as np
import pandas as pd
from gem.mpa import MetaphlanProfileExtractor, MetaphlanProfile


def select_profiles_in_metadata(metadata_df, profiles_df) -> pd.DataFrame:
    return profiles_df.loc[profiles_df.index.isin(metadata_df['Sample ID'])]


def main(
        profile_tsv_path: Path,
        metadata_tsv_path: Path,
        train_out_path: Path,
        test_out_path: Path,
        edge_weight_strategy: str,
        optional_newick_tree_path: Optional[Path] = None,
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
    print("Number of samples (Adult & Healthy): {}".format(metadata.shape[0]))

    if edge_weight_strategy == "jaccard":
        similarity = JaccardSimilarityOracle()
    elif edge_weight_strategy == "phylogenetic":
        tree = Phylo.read(optional_newick_tree_path, "newick")
        similarity = PhylogeneticSimilarityOracle(tree)
    else:
        raise ValueError(f"Unrecognized edge_weight_strategy option `{edge_weight_strategy}")
    train_df, test_df = test_train_split_asv_separation(profiles_indexed, metadata_subset, similarity)

    train_df.to_csv(train_out_path, sep="\t", index=True)
    test_df.to_csv(test_out_path, sep="\t", index=True)

    print("# train samples: {}".format(train_df.shape[0]))
    print("# test samples: {}".format(test_df.shape[0]))
    print("Ratio: {} / {} = {}".format(
        train_df.shape[0], test_df.shape[0], train_df.shape[0] / test_df.shape[0]
    ))


def jaccard_similarity(x: Set, y: Set) -> float:
    numer = len(x.intersection(y))
    denom = len(x.union(y))
    return numer / denom


class SampleSimilarityOracle(ABC):
    @abstractmethod
    def similarity(self, x: MetaphlanProfile, y: MetaphlanProfile) -> float:
        pass


class JaccardSimilarityOracle(SampleSimilarityOracle):
    def similarity(self, x: MetaphlanProfile, y: MetaphlanProfile) -> float:
        return jaccard_similarity(set(x.sgb_ids), set(y.sgb_ids))


class PhylogeneticSimilarityOracle(SampleSimilarityOracle):
    SGB_PREFIX_LEN = len("SGB")

    def __init__(self, tree: Tree):
        self.tree = tree

    def similarity(self, x: MetaphlanProfile, y: MetaphlanProfile) -> float:
        pairwise_distances = np.array([
            [
                self.tree.distance(x_sgb[self.SGB_PREFIX_LEN:], y_sgb[self.SGB_PREFIX_LEN:])
                for y_sgb in y.sgb_ids
            ]
            for x_sgb in x.sgb_ids
        ], dtype=float)

        x_nn_dist = np.min(pairwise_distances, axis=1)  # nearest neighbor dist for each SGB in x
        y_nn_dist = np.min(pairwise_distances, axis=0)  # nearest neighbor dist for each SGB in y

        sym_nn_dist = 0.5 * (np.mean(x_nn_dist) + np.mean(y_nn_dist))  # symmetrized average of both
        return np.exp(-sym_nn_dist)  # kernelized value, converting distance to similarity.


def test_train_split_asv_separation(
        profiles_indexed: pd.DataFrame,
        metadata_subset_df: pd.DataFrame,
        similarity_oracle: SampleSimilarityOracle,
):
    """ Main idea: Train samples should not share any ASVs with test samples. """
    import networkx as nx
    from scipy.linalg import eigh

    def cut_edge_weights(G: nx.Graph, partition1: List, partition2: List):
        """Calculate the cut value between two partitions"""
        cut_values = []
        for i in partition1:
            for j in partition2:
                if G.has_edge(i, j):
                    cut_values.append(G[i][j].get('weight', 0.0))
        return cut_values

    def spectral_division(G: nx.Graph, left_q: float, right_q: float):
        """
        Computes an embedding of samples via spectral decomposition of a certain graph. Nodes are samples, edge weights are Jaccard Similarity.
        After the embedding is computed, computes the left-tail and right-tail set of nodes to assign (specified via quantile "q" values).
        """
        if not nx.is_connected(G):
            raise Exception("Graph not connected!")

        # Get Laplacian matrix0
        node_order = list(G.nodes())
        L_sparse = nx.laplacian_matrix(G, weight='weight', nodelist=node_order)
        L_dense = L_sparse.todense()  # Or use L.toarray() for a NumPy ndarray

        # Find eigenvalues and eigenvectors
        print("computing eigendecomposition.")
        # eigenvals, eigenvecs = eigsh(L_sparse, k=2, which='SM')
        eigenvals, eigenvecs = eigh(L_dense, subset_by_index=(0, 1))
        print("eigenvalues:", eigenvals)

        # Second smallest eigenvalue and its eigenvector (Fiedler vector)
        fiedler_vector = eigenvecs[:, 1]

        # Assign left subset and right subset using the input parameters.
        left_ub = np.quantile(fiedler_vector, q=left_q)
        right_lb = np.quantile(fiedler_vector, q=right_q)
        print("Using tail cutoff quantiles q_left = {}, q_right = {}".format(left_q, right_q))
        partition_left = [node_order[i] for i in range(len(node_order)) if fiedler_vector[i] < left_ub]
        partition_right = [node_order[i] for i in range(len(node_order)) if fiedler_vector[i] >= right_lb]
        cut_w = cut_edge_weights(G, partition_left, partition_right)
        print("Partition is {} vs. {} [Cut weights: mean={}, median={}, max={}, min={}]".format(
            len(partition_left), len(partition_right),
            np.mean(cut_w), np.median(cut_w), np.max(cut_w), np.min(cut_w)
        ))
        # plt.hist(cut_w, bins=20)
        return partition_left, partition_right

    def weighted_graph_from_matrix(A: np.ndarray, node_names: List[str]):
        n = A.shape[0]
        G = nx.Graph()
        G.add_nodes_from(node_names)
        for i, j in itertools.combinations(range(n), 2):
            if A[i, j] > 0:
                G.add_edge(node_names[i], node_names[j], weight=A[i, j])
        return G

    def train_test_sgb_jaccard_spectral_split(profile_df: pd.DataFrame, train_fraction: float, test_fraction: float):
        extractor = MetaphlanProfileExtractor(profile_df)
        all_samples = list(extractor.samples())
        print(f"Splitting {len(all_samples)} samples found in project.")

        # Compute weighted adjacency matrix, A[i,j] = # of SGBs shared by sample i and j.
        n_samples = len(all_samples)
        n_pairs = int(n_samples * (n_samples - 1) / 2)
        A = np.zeros((n_samples, n_samples), dtype=float)
        for (i, sample_i), (j, sample_j) in tqdm(
                itertools.combinations(enumerate(all_samples), r=2),
                total=n_pairs,
                desc="Sample pair calculation",
        ):
            sample_sgb_sim = similarity_oracle.similarity(sample_i, sample_j)
            A[i, j] = sample_sgb_sim
            A[j, i] = sample_sgb_sim
        print("Computed sample similarity matrix of shape {}.".format(A.shape))

        # Create a graph, using "A" as an adjacency matrix. Enumerate all connected components.
        # first rule out all small/trivial connected components.
        G = weighted_graph_from_matrix(A, [sample.sample_id for sample in all_samples])
        ccs = list(nx.connected_components(G))
        ccs = sorted(ccs, key=lambda x: len(x), reverse=True)

        training_sample_ids = []
        test_sample_ids = []
        print("# SGB-connected components: {}".format(len(ccs)))
        for cc in ccs:
            print("component sample count = {}".format(len(cc)))
            if len(cc) <= 5:
                training_sample_ids += list(cc)
            else:
                G_cc = G.subgraph(cc).copy()
                test_samples, train_samples = spectral_division(G_cc, left_q=test_fraction, right_q=1 - train_fraction)
                training_sample_ids += train_samples
                test_sample_ids += test_samples

        training_sample_ids = set(training_sample_ids)
        test_sample_ids = set(test_sample_ids)

        train_df = profile_df[profile_df.index.isin(training_sample_ids)]
        test_df = profile_df[profile_df.index.isin(test_sample_ids)]
        return train_df, test_df

    train_df, test_df = train_test_sgb_jaccard_spectral_split(
        select_profiles_in_metadata(metadata_subset_df, profiles_indexed),
        train_fraction=0.4, test_fraction=0.1
    )
    return train_df, test_df


if __name__ == "__main__":
    DATA_DIR = Path("/data/cctm/youn/metaphlan_dset/dataset")
    OUT_DIR = Path("/data/cctm/youn/metaphlan_dset/model_training")

    full_profile_tsv = DATA_DIR / "BlancoMiguezA_2023_profiles.tsv"
    metadata_tsv = DATA_DIR / "BlancoMiguezA_2023_metadata.tsv"
    tree_path = Path("/data/cctm/youn/metaphlan_dset/database/mpa_vJan21_CHOCOPhlAnSGB_202103.nwk")
    train_out = OUT_DIR / "train.tsv"
    test_out = OUT_DIR / "test.tsv"

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    main(
        profile_tsv_path=full_profile_tsv,
        metadata_tsv_path=metadata_tsv,
        train_out_path=train_out,
        test_out_path=test_out,
        edge_weight_strategy="phylogenetic",
        optional_newick_tree_path=tree_path,
    )
