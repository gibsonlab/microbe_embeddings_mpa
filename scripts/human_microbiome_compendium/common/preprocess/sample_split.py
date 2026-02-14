""" Option 3: Train samples should not share any ASVs with test samples. """
from typing import *
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eigh
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

from ..ml import *
from ..util import ASVDistanceMatrix


def distance_matrix_to_similarity_matrix(mat: ASVDistanceMatrix) -> np.ndarray:
    mat = mat.matrix
    assert len(mat.shape) == 2, "Input must be a 2-d matrix."
    assert mat.shape[0] == mat.shape[1], "Input must be a square matrix, got: {}".format(mat.shape)
    d_max = np.max(mat)
    return 1 - (1/d_max) * mat


def cut_edge_weights(G: nx.Graph, partition1: List, partition2: List):
    """Calculate the cut value between two partitions"""
    cut_values = []
    for i in partition1:
        for j in partition2:
            if G.has_edge(i, j):
                cut_values.append(G[i][j].get('weight', 0.0))
    return cut_values


def jaccard_similarity(x: Set, y: Set) -> float:
    numer = len(x.intersection(y))
    denom = len(x.union(y))
    return numer / denom


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
    plt.hist(cut_w, bins=20)
    return partition_left, partition_right


def weighted_graph_from_matrix(A: np.ndarray, node_names: List[str]):
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(node_names)
    for i, j in itertools.combinations(range(n), 2):
        if A[i, j] > 0:
            G.add_edge(node_names[i], node_names[j], weight=A[i, j])
    return G


def train_test_split(
        sample_df: pd.DataFrame,
        train_fraction: float,
        test_fraction: float,
        method: str,
        *args,
        **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
    if method == 'random':
        print("Using random splitting.")
        train, test = train_test_split_random(sample_df, train_fraction, test_fraction, *args, **kwargs)
        return train, test, None
    elif method == 'spectral-mincut':
        print("Using Min-cut splitting.")
        train, test = train_test_split_mincut_approximation(sample_df, train_fraction, test_fraction, *args, **kwargs)
        return train, test, None
    elif method == 'pcoa':
        print("Using PCOA-based splitting.")
        return train_test_split_pcoa(sample_df, train_fraction, test_fraction, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported method `{method}`.")


def train_test_split_random(
        sample_df: pd.DataFrame,
        train_fraction: float,
        test_fraction: float,
        seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_df.shape[0])
    rng.shuffle(indices)

    # split the indices.
    n_train_rows = int(sample_df.shape[0] * train_fraction)
    n_test_rows = int(sample_df.shape[0] * test_fraction)
    train_indices = indices[:n_train_rows]
    test_indices = indices[n_train_rows:n_train_rows + n_test_rows]

    # gather the rows.
    train_df = sample_df.iloc[train_indices].reset_index(drop=True)
    test_df = sample_df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df


def train_test_split_pcoa(
        sample_df: pd.DataFrame,
        train_fraction: float,
        test_fraction: float,
        sample_dist_mat_order: List[str],
        sample_distance_matrix: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import skbio
    import seaborn as sb
    pcoa_result = skbio.stats.ordination.pcoa(sample_distance_matrix, method='eigh')

    """ PCoA based coordinate plotting of samples """
    coordinates: pd.DataFrame = pcoa_result.samples[['PC1', 'PC2']].assign(SampleId=sample_dist_mat_order)
    pc1 = coordinates['PC1'].to_numpy()
    left_q = train_fraction
    right_q = 1 - test_fraction

    # Assign left subset and right subset using the input parameters.
    left_ub = np.quantile(pc1, q=left_q)
    right_lb = np.quantile(pc1, q=right_q)
    print("Using tail cutoff quantiles q_left = {}, q_right = {}".format(left_q, right_q))
    partition_left = [sample_id for i, sample_id in enumerate(sample_dist_mat_order) if pc1[i] < left_ub]
    partition_right = [sample_id for i, sample_id in enumerate(sample_dist_mat_order) if pc1[i] >= right_lb]

    print("Partition is {} vs. {}".format(len(partition_left), len(partition_right)))
    training_sample_ids = set(partition_left)
    test_sample_ids = set(partition_right)
    train_df = sample_df.loc[sample_df['srs'].isin(training_sample_ids)]
    test_df = sample_df.loc[sample_df['srs'].isin(test_sample_ids)]

    # ============================ plot output.
    train_sample_set = set(train_df['srs'])
    test_sample_set = set(test_df['srs'])

    test_train_labels = []
    for sample_id in sample_dist_mat_order:
        if sample_id in train_sample_set:
            test_train_labels.append("Train")
        elif sample_id in test_sample_set:
            test_train_labels.append("Test")
        else:
            test_train_labels.append("Excluded")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sb.scatterplot(
        coordinates.assign(Label=test_train_labels),
        x='PC1', y='PC2', hue="Label",
        alpha=0.3, linewidth=0., axis=ax
    )

    # Get proportion of variance explained
    prop_var = pcoa_result.proportion_explained
    ax.set_xlabel(f'PC1 ({prop_var[0] * 100:.2f}%)')
    ax.set_ylabel(f'PC2 ({prop_var[1] * 100:.2f}%)')
    ax.set_title('PCoA Plot')
    ax.grid(True, alpha=0.3)
    plt.show()
    return train_df, test_df, coordinates


def train_test_split_mincut_approximation(
        sample_df: pd.DataFrame,
        train_fraction: float,
        test_fraction: float,
        sample_dist_mat_order: List[str],
        sample_distance_matrix: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits samples by applying spectral cut algorithm to each CC.
    """
    # Collect the list of samples across all projects.

    print(f"[MINCUT split] Splitting {len(sample_dist_mat_order)} samples")

    assert sample_distance_matrix.shape[0] == sample_distance_matrix.shape[1], "Distance matrix must be a square matrix."
    assert sample_distance_matrix.shape[0] == len(sample_dist_mat_order), "Distance matrix n_rows must match dimensions of the number of samples in collection."
    print("Using pre-computed sample-to-sample distance matrix.")
    print("Using conversion: Similarity = (1 - d / d_max)")
    A = 1 - (sample_distance_matrix / np.max(sample_distance_matrix))
    print("Computed sample similarity matrix of shape {}.".format(A.shape))

    # Create a graph, using "A" as an adjacency matrix. Enumerate all connected components.
    # first rule out all small/trivial connected components.
    G = weighted_graph_from_matrix(A, sample_dist_mat_order)
    ccs = list(nx.connected_components(G))
    ccs = sorted(ccs, key=lambda x: len(x), reverse=True)

    training_sample_ids = []
    test_sample_ids = []
    print("# ASV-connected components: {}".format(len(ccs)))
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
    train_df = sample_df.loc[sample_df['srs'].isin(training_sample_ids)]
    test_df = sample_df.loc[sample_df['srs'].isin(test_sample_ids)]
    return train_df, test_df