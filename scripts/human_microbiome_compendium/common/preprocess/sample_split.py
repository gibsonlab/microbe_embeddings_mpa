""" Option 3: Train samples should not share any ASVs with test samples. """
from typing import *
from pathlib import Path
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from ..ml import *


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


def spectral_division(G: nx.Graph, left_q: float, right_q: float, cc_name: str):
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
    plt.hist(cut_w, bins=20, alpha=0.6, label=cc_name)
    return partition_left, partition_right


def weighted_graph_from_matrix(A: np.ndarray, node_names: List[str]):
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(node_names)
    for i, j in itertools.combinations(range(n), 2):
        if A[i, j] > 0:
            G.add_edge(node_names[i], node_names[j], weight=A[i,j])
    return G


def train_test_split_mincut_approximation(
    sample_df: pd.DataFrame, 
    abundance_table_dir: Path,
    asv_id_subset: Set[str], 
    train_fraction: float, 
    test_fraction: float
):
    # Collect the list of samples across all projects.
    proj_ids = list(pd.unique(sample_df['project']))
    all_samples = []
    for proj_id, proj_section in sample_df.groupby('project'):
        proj = MicrobiomeProject(proj_id, abundance_table_dir, sample_df)
        proj_sample_subset_ids = set(proj_section['srs'])
        all_samples = all_samples + [s for s in proj.samples if s.sample_id in proj_sample_subset_ids]
    
    print(f"Splitting {len(all_samples)} samples found in projects: {proj_ids}")
    
    # Compute weighted adjacency matrix, A[i,j] = # of ASVs shared by sample i and j.
    n_samples = len(all_samples)
    n_pairs = int(n_samples * (n_samples - 1) / 2)
    A = np.zeros((n_samples, n_samples), dtype=float)
    for (i, sample_i), (j, sample_j) in tqdm(
        itertools.combinations(enumerate(all_samples), r=2),
        total=n_pairs,
        desc="Sample pair calculation",
    ):
        sample_asv_sim = jaccard_similarity(sample_i.asv_ids, sample_j.asv_ids)
        A[i, j] = sample_asv_sim
        A[j, i] = sample_asv_sim
    print("Computed sample similarity matrix of shape {}.".format(A.shape))
    
    # Create a graph, using "A" as an adjacency matrix. Enumerate all connected components.
    # first rule out all small/trivial connected components.
    G = weighted_graph_from_matrix(A, [sample.sample_id for sample in all_samples])
    ccs = list(nx.connected_components(G))
    ccs = sorted(ccs, key=lambda x: len(x), reverse=True)

    training_sample_ids = []
    test_sample_ids = []
    print("# ASV-connected components: {}".format(len(ccs)))
    for cc_idx, cc in enumerate(ccs):
        print("component sample count = {}".format(len(cc)))
        if len(cc) <= 5:
            training_sample_ids += list(cc)
        else:
            G_cc = G.subgraph(cc).copy()
            test_samples, train_samples = spectral_division(
                G_cc, left_q=test_fraction, right_q=1-train_fraction,
                cc_name=f"CC_{cc_idx}"
            )
            training_sample_ids += train_samples
            test_sample_ids += test_samples
    plt.legend()
    plt.show()

    training_sample_ids = set(training_sample_ids)
    test_sample_ids = set(test_sample_ids)
    train_df = sample_df.loc[sample_df['srs'].isin(training_sample_ids)]
    test_df = sample_df.loc[sample_df['srs'].isin(test_sample_ids)]
    return train_df, test_df