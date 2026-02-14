from abc import abstractmethod, ABC
from typing import *
from pathlib import Path
import itertools

from skbio.stats.ordination import pcoa
from tqdm import tqdm

from joblib import Parallel, delayed

import numpy as np
from numba import njit, prange
import pandas as pd
from gem.datasets.abundance_profile import MetaphlanProfileParser, MetaphlanProfile



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

    # if edge_weight_strategy == "jaccard":
    #     similarity = JaccardSimilarityOracle()
    # elif edge_weight_strategy == "phylogenetic":
    #     similarity = PhylogeneticSimilarityOracle(optional_newick_tree_path)
    # else:
    #     raise ValueError(f"Unrecognized edge_weight_strategy option `{edge_weight_strategy}")
    # train_df, test_df = test_train_split_asv_separation(profiles_indexed, metadata_subset, similarity)
    pcoa_plot_path = train_out_path.parent / "pcoa_plot.png"
    train_df, test_df = test_train_split_pcoa_jensenshannon(profiles_indexed, metadata_subset, plot_path=pcoa_plot_path)

    train_df.to_csv(train_out_path, sep="\t", index=True)
    test_df.to_csv(test_out_path, sep="\t", index=True)

    print("# train samples: {}".format(train_df.shape[0]))
    print("# test samples: {}".format(test_df.shape[0]))
    print("Ratio: {} / {} = {}".format(
        train_df.shape[0], test_df.shape[0], train_df.shape[0] / test_df.shape[0]
    ))


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
        plot_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profile_df = select_profiles_in_metadata(metadata_subset_df, profiles_indexed)
    extractor = MetaphlanProfileParser(profile_df)

    abundances = extractor.sgb_profile_df.to_numpy()
    abundances = abundances / abundances.sum(axis=-1, keepdims=True)
    sample_ids = [str(sid) for sid in extractor.sgb_profile_df.index]

    print("Calculating jensen-shannon distance matrix.")
    dist_mat = calculate_js_distances_numba(abundances)

    from skbio import DistanceMatrix
    dist_mat = DistanceMatrix(data=dist_mat, ids=sample_ids)
    pcoa_result = pcoa(dist_mat, method='eigh')
    # noinspection PyTypeHints
    coordinates = pcoa_result.samples.loc[:, ['PC1', 'PC2']].assign(SampleId=sample_ids)
    pc1 = coordinates['PC1'].to_numpy()
    left_q = train_fraction
    right_q = 1 - test_fraction

    # Assign left subset and right subset using the input parameters.
    left_ub = np.quantile(pc1, q=left_q)
    right_lb = np.quantile(pc1, q=right_q)
    print("Using tail cutoff quantiles train < {}, test >= {}".format(left_q, right_q))
    partition_left = [sample_id for i, sample_id in enumerate(sample_ids) if pc1[i] < left_ub]
    partition_right = [sample_id for i, sample_id in enumerate(sample_ids) if pc1[i] >= right_lb]

    print("Partition is {} vs. {}".format(len(partition_left), len(partition_right)))
    training_sample_ids = set(partition_left)
    test_sample_ids = set(partition_right)
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

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sb.scatterplot(
            coordinates.assign(Label=test_train_labels),
            x='PC1', y='PC2', hue="Label",
            alpha=0.3, linewidth=0., ax=ax
        )

        # Get proportion of variance explained
        prop_var = pcoa_result.proportion_explained
        ax.set_xlabel(f'PC1 ({prop_var[0] * 100:.2f}%)')
        ax.set_ylabel(f'PC2 ({prop_var[1] * 100:.2f}%)')
        ax.set_title('PCoA Plot')
        ax.grid(True, alpha=0.3)
        plt.savefig(plot_path, bbox_inches='tight')
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



# ================================================= HELPER CODE: fast distance calculation in newick-formatted phylo tree

class TreeNode:
    """Represents a node in the phylogenetic tree."""
    def __init__(self, name: Optional[str] = None, node_id: int = -1):
        self.name = name
        self.node_id = node_id
        self.children: List[TreeNode] = []
        self.parent: Optional[TreeNode] = None
        self.branch_length: float = 0.0
        self.dist_from_root: float = 0.0
        self.depth: int = 0  # Depth in tree (for LCA)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

class NewickParser:
    """Parser for Newick format phylogenetic trees."""
    def __init__(self, newick_string: str):
        self.newick = newick_string.strip()
        self.pos = 0
        self.node_counter = 0

    def parse(self) -> TreeNode:
        """Parse the Newick string and return the root node."""
        if self.newick.endswith(';'):
            self.newick = self.newick[:-1]
        root = self._parse_node()
        return root

    def _parse_node(self) -> TreeNode:
        """Recursively parse a node and its descendants."""
        node = TreeNode(node_id=self.node_counter)
        self.node_counter += 1
        if self.pos < len(self.newick) and self.newick[self.pos] == '(':
            # Internal node with children
            self.pos += 1  # skip '('
            while True:
                child = self._parse_node()
                child.parent = node
                node.children.append(child)
                if self.pos < len(self.newick) and self.newick[self.pos] == ',':
                    self.pos += 1  # skip ','
                else:
                    break
            if self.pos < len(self.newick) and self.newick[self.pos] == ')':
                self.pos += 1  # skip ')'

        # Parse node label
        label_start = self.pos
        while self.pos < len(self.newick) and self.newick[self.pos] not in ',:();':
            self.pos += 1
        if self.pos > label_start:
            node.name = self.newick[label_start:self.pos]

        # Parse branch length
        if self.pos < len(self.newick) and self.newick[self.pos] == ':':
            self.pos += 1  # skip ':'
            length_start = self.pos
            while self.pos < len(self.newick) and self.newick[self.pos] not in ',();':
                self.pos += 1
            length_str = self.newick[length_start:self.pos]
            try:
                node.branch_length = float(length_str)
            except ValueError:
                node.branch_length = 0.0
        return node

class OptimizedDistanceMatrixComputer:
    """
    Optimized distance matrix computation using efficient LCA queries.
    Key optimizations:
    1. Precompute all distances from root
    2. Use binary lifting for O(log H) LCA queries (H = height)
    3. Batch process columns to improve cache locality
    """
    def __init__(self, newick_file: Path):
        """Initialize with a Newick format tree string."""
        print(f"Parsing tree file {newick_file}")
        with open(newick_file, 'r') as f:
            newick_string = f.read().strip()
        parser = NewickParser(newick_string)
        self.root = parser.parse()
        self.leaf_nodes: Dict[str, TreeNode] = {}
        self.all_nodes: List[TreeNode] = []
        # Collect all nodes and leaves
        print("collecting all nodes and leaves...")
        self._collect_nodes(self.root)
        # Precompute distances and depths
        print("precomputing distances and tree depths...")
        self._compute_root_distances_and_depths(self.root, 0.0, 0)
        # Precompute binary lifting table for fast LCA
        print("precomputing binary lifting table for fast LCA lookup...")
        self._precompute_lca_table()

    def _collect_nodes(self, node: TreeNode) -> None:
        """Collect all nodes and leaf nodes."""
        self.all_nodes.append(node)
        if node.is_leaf():
            if node.name:
                self.leaf_nodes[node.name] = node
        else:
            for child in node.children:
                self._collect_nodes(child)

    def _compute_root_distances_and_depths(self, node: TreeNode, dist: float, depth: int) -> None:
        """Compute distance from root and depth for all nodes."""
        node.dist_from_root = dist
        node.depth = depth
        for child in node.children:
            self._compute_root_distances_and_depths(
                child,
                dist + child.branch_length,
                depth + 1
            )

    def _precompute_lca_table(self) -> None:
        """
        Precompute binary lifting table for O(log H) LCA queries.
        ancestor[node][i] = 2^i-th ancestor of node
        """
        n = len(self.all_nodes)
        if n == 0:
            return
        # Find maximum depth to determine table size
        max_depth = max(node.depth for node in self.all_nodes)
        log_depth = max_depth.bit_length()
        # Initialize table
        self.ancestor = {}
        for node in self.all_nodes:
            self.ancestor[node.node_id] = [None] * log_depth
        # Fill table
        for node in self.all_nodes:
            if node.parent is not None:
                self.ancestor[node.node_id][0] = node.parent
        # Binary lifting: ancestor[node][i] = ancestor[ancestor[node][i-1]][i-1]
        for i in range(1, log_depth):
            for node in self.all_nodes:
                if self.ancestor[node.node_id][i-1] is not None:
                    prev_ancestor = self.ancestor[node.node_id][i-1]
                    self.ancestor[node.node_id][i] = self.ancestor[prev_ancestor.node_id][i-1]

    def _find_lca_optimized(self, node1: TreeNode, node2: TreeNode) -> TreeNode:
        """
        Find LCA using binary lifting in O(log H) time.
        """
        # Make node1 the deeper node
        if node1.depth < node2.depth:
            node1, node2 = node2, node1
        # Bring node1 to same level as node2
        depth_diff = node1.depth - node2.depth
        log_depth = len(self.ancestor[node1.node_id])
        for i in range(log_depth):
            if depth_diff & (1 << i):
                node1 = self.ancestor[node1.node_id][i]
                if node1 is None:
                    break
        # If node2 is ancestor of node1
        if node1 == node2:
            return node1
        # Binary search for LCA
        for i in range(log_depth - 1, -1, -1):
            if (self.ancestor[node1.node_id][i] is not None and
                self.ancestor[node2.node_id][i] is not None and
                self.ancestor[node1.node_id][i] != self.ancestor[node2.node_id][i]):
                node1 = self.ancestor[node1.node_id][i]
                node2 = self.ancestor[node2.node_id][i]
        return node1.parent if node1.parent is not None else node1

    def compute_distance_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Compute L×K distance matrix efficiently.
        Time Complexity: O(L*log(L) + K*L*log(H)) where H is tree height
        Space Complexity: O(N*log(H)) for LCA table, where N is total nodes
        For balanced trees, H = O(log(L)), giving O(L*log(L) + K*L*log(log(L)))
        For unbalanced trees, worst case is still better than naive O(LK) with large constants.
        """
        all_leaves = sorted(self.leaf_nodes.keys())
        L = len(all_leaves)
        # Verify all subset leaves exist
        for leaf in all_leaves:
            if leaf not in self.leaf_nodes:
                raise ValueError(f"Leaf '{leaf}' not found in tree")
        # Initialize distance matrix
        dist_matrix = np.zeros((L, L), dtype=float)
        # Precompute leaf nodes for all_leaves
        all_leaf_nodes = [self.leaf_nodes[leaf] for leaf in all_leaves]

        # For each column (target leaf in subset)
        for k, target_node in enumerate(all_leaf_nodes):
            # Compute distances from target to all leaves
            for i, source_node in enumerate(all_leaf_nodes):
                if i <= k:
                    continue
                else:
                    # Find LCA and compute distance
                    lca = self._find_lca_optimized(source_node, target_node)
                    dist = (source_node.dist_from_root + target_node.dist_from_root - 2 * lca.dist_from_root)
                    dist_matrix[i, k] = dist
                    dist_matrix[k, i] = dist
        return dist_matrix, all_leaves

    def get_all_leaf_names(self) -> List[str]:
        """Return sorted list of all leaf names."""
        return sorted(self.leaf_nodes.keys())


# ======================================================================================


class SampleSimilarityOracle(ABC):
    @abstractmethod
    def similarity(self, x: MetaphlanProfile, y: MetaphlanProfile) -> float:
        pass


class JaccardSimilarityOracle(SampleSimilarityOracle):
    def similarity(self, x: MetaphlanProfile, y: MetaphlanProfile) -> float:
        return jaccard_similarity(set(x.sgb_ids), set(y.sgb_ids))


class PhylogeneticSimilarityOracle(SampleSimilarityOracle):
    def __init__(self, newick_path: Path):
        computer = OptimizedDistanceMatrixComputer(newick_path)
        dist_matrix, leaf_order = computer.compute_distance_matrix()

        self.dist_matrix = dist_matrix
        self.leaf_indices = {
            f'SGB{sgb_id}': idx
            for idx, sgb_id in enumerate(leaf_order)
        }

    def sgb_dist(self, sgb_id1: str, sgb_id2: str) -> float:
        i = self.leaf_indices[sgb_id1]
        j = self.leaf_indices[sgb_id2]
        return self.dist_matrix[i, j]

    def similarity(self, x: MetaphlanProfile, y: MetaphlanProfile) -> float:
        x_sgb_ids = [sgb_id for sgb_id in x.sgb_ids if sgb_id in self.leaf_indices]
        y_sgb_ids = [sgb_id for sgb_id in y.sgb_ids if sgb_id in self.leaf_indices]
        if len(x_sgb_ids) == 0:
            raise Exception(f"Sample {x.sample_id} had no SGBs to be found inside the tree.")
        if len(y_sgb_ids) == 0:
            raise Exception(f"Sample {y.sample_id} had no SGBs to be found inside the tree.")

        pairwise_distances = np.array([
            [
                self.sgb_dist(x_sgb, y_sgb)
                for y_sgb in y_sgb_ids
            ]
            for x_sgb in x_sgb_ids
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

    # helpers for joblib -- parallelization
    def compute_similarity(args):
        (i, sample_i), (j, sample_j) = args
        return i, j, similarity_oracle.similarity(sample_i, sample_j)

    def train_test_sgb_jaccard_spectral_split(profile_df: pd.DataFrame, train_fraction: float, test_fraction: float):
        extractor = MetaphlanProfileParser(profile_df)
        all_samples = list(extractor.samples())
        print(f"Splitting {len(all_samples)} samples found in project.")

        # Compute weighted adjacency matrix, A[i,j] = # of SGBs shared by sample i and j.
        n_samples = len(all_samples)
        # n_pairs = int(n_samples * (n_samples - 1) / 2)
        A = np.zeros((n_samples, n_samples), dtype=float)

        print("Using joblib to compute sample pair similarities in parallel.")
        pairs = list(itertools.combinations(enumerate(all_samples), r=2))
        results = Parallel(n_jobs=-1)(
            delayed(compute_similarity)(pair)
            for pair in tqdm(pairs, desc="Sample pair calculation")
        )

        for i, j, sim in results:
            A[i, j] = sim
            A[j, i] = sim

        # for (i, sample_i), (j, sample_j) in tqdm(
        #         itertools.combinations(enumerate(all_samples), r=2),
        #         total=n_pairs,
        #         desc="Sample pair calculation",
        # ):
        #     sample_sgb_sim = similarity_oracle.similarity(sample_i, sample_j)
        #     A[i, j] = sample_sgb_sim
        #     A[j, i] = sample_sgb_sim
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
    OUT_DIR = Path("/data/cctm/youn/metaphlan_dset/model_training_pcoa_split")
    print(f"Destination OUT_DIR: {OUT_DIR}")

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
