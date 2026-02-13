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


def train_test_split_mincut_approximation(
        sample_df: pd.DataFrame,
        abundance_table_dir: Path,
        asv_id_subset: Set[str],
        train_fraction: float,
        test_fraction: float,
        sample_distance_matrix: Optional[np.ndarray] = None,
        asv_distance_matrix: Optional[ASVDistanceMatrix] = None,
):
    """
    Splits samples by applying spectral cut algorithm to each CC.
    """
    # Collect the list of samples across all projects.
    proj_ids = list(pd.unique(sample_df['project']))
    all_samples: List[MicrobiomeSample] = []
    for proj_id, proj_section in sample_df.groupby('project'):
        proj = MicrobiomeProject(str(proj_id), abundance_table_dir, sample_df)
        proj_sample_subset_ids = set(proj_section['srs'])
        for sample in proj.samples:
            if sample.sample_id in proj_sample_subset_ids:
                n_sample_asvs = len([asv_id for asv_id in sample.asv_ids if asv_id in asv_id_subset])
                if n_sample_asvs > 0:
                    all_samples.append(sample)

    print(f"[FAIR SPLIT] Splitting {len(all_samples)} samples found in projects: {proj_ids}")

    if sample_distance_matrix is not None:
        assert sample_distance_matrix.shape[0] == sample_distance_matrix.shape[1], "Distance matrix must be a square matrix."
        assert sample_distance_matrix.shape[0] == len(all_samples), "Distance matrix n_rows must match dimensions of the number of samples in collection."
        print("Using pre-computed sample-to-sample distance matrix.")
        print("Using conversion: Similarity = (1 - d / d_max)")
        A = 1 - (sample_distance_matrix / np.max(sample_distance_matrix))
        print("Computed sample similarity matrix of shape {}.".format(A.shape))
    else:
        print("Computing sample-to-sample distance matrix from ASV collections.")
        # calculate sample distance matrix.
        # Compute weighted adjacency matrix, A[i,j] = # of ASVs shared by sample i and j.
        n_samples = len(all_samples)
        n_pairs = int(n_samples * (n_samples - 1) / 2)
        A = np.zeros((n_samples, n_samples), dtype=float)
        if asv_distance_matrix is not None:
            print("Weighted Graph will use weight = (1 - d/d_max) as similarity metric.")

            def sim_fn(sample1: MicrobiomeSample, sample2: MicrobiomeSample):
                distance_submat = asv_distance_matrix.submatrix(
                    [asv_id for asv_id in sample1.asv_ids if asv_id in asv_id_subset],
                    [asv_id for asv_id in sample2.asv_ids if asv_id in asv_id_subset],
                )
                if distance_submat.size == 0:
                    raise ValueError(
                        "Distance submatrix had size 0. Some sample (illegally) had 0 ASVs in filtered collection.")
                else:
                    sim_submat = 1 - (distance_submat / asv_distance_matrix.seq_len())  # SIMILARITY = 1 - HAMMING_DIST / SEQLEN
                    return np.mean(sim_submat)
        else:
            print("Weighted Graph will use weight = JACCARD(i,j) as similarity metric.")
            sim_fn = lambda sample1, sample2: jaccard_similarity(
                sample1.asv_ids.intersection(asv_id_subset),
                sample2.asv_ids.intersection(asv_id_subset)
            )

        for (i, sample_i), (j, sample_j) in tqdm(
                itertools.combinations(enumerate(all_samples), r=2),
                total=n_pairs,
                desc="Sample pair calculation",
        ):
            sample_asv_sim = sim_fn(sample_i, sample_j)
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