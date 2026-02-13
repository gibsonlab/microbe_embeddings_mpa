# """ Option 3: Train samples should not share any ASVs with test samples. """
# from typing import *
# from pathlib import Path
# import itertools
#
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# import networkx as nx
# from scipy.linalg import eigh
# import matplotlib.pyplot as plt
#
# from ..ml import *
# from ..util import ASVDistanceMatrix
#
#
# def cut_edge_weights(G: nx.Graph, set1: Set, set2: Set) -> List[float]:
#     """Calculate the cut value between two partitions"""
#     cut_values = []
#     for i in set1:
#         for j in set2:
#             if G.has_edge(i, j):
#                 cut_values.append(G[i][j]['weight'])
#             else:
#                 cut_values.append(0.0)
#     return cut_values
#
#
# def jaccard_similarity(x: Set, y: Set) -> float:
#     numer = len(x.intersection(y))
#     denom = len(x.union(y))
#     return numer / denom
#
#
# def weighted_graph_from_matrix(A: np.ndarray, node_names: List[str]):
#     n = A.shape[0]
#     G = nx.Graph()
#     G.add_nodes_from(node_names)
#     for i, j in itertools.combinations(range(n), 2):
#         if A[i, j] > 0:
#             G.add_edge(node_names[i], node_names[j], weight=A[i,j])
#     return G
#
#
# def split_component_spectral(
#         G: nx.Graph,
#         size_A: int,
#         size_B: int
# ) -> Tuple[Set[Any], Set[Any]]:
#     """
#     Split a connected component using spectral method.
#
#     Args:
#         subgraph: NetworkX subgraph of the component
#         size_A: Number of nodes needed for A
#         size_B: Number of nodes needed for B
#
#     Returns:
#         A_nodes: Nodes assigned to A
#         B_nodes: Nodes assigned to B
#     """
#     # Get Laplacian matrix0
#     node_order = list(G.nodes())
#     L_sparse = nx.laplacian_matrix(G, weight='weight', nodelist=node_order)
#     L_dense = L_sparse.todense()  # Or use L.toarray() for a NumPy ndarray
#
#     # Find eigenvalues and eigenvectors
#     print("computing eigendecomposition.")
#     # eigenvals, eigenvecs = eigsh(L_sparse, k=2, which='SM')
#     eigenvals, eigenvecs = eigh(L_dense, subset_by_index=(0, 1))
#     print("eigenvalues:", eigenvals)
#
#     # Second smallest eigenvalue and its eigenvector (Fiedler vector)
#     fiedler_vector = eigenvecs[:, 1]
#
#     # Sort nodes by Fiedler vector value
#     sorted_indices = np.argsort(fiedler_vector)
#
#     # Assign nodes: try to minimize cut by keeping similar values together
#     # Put lowest values in A, highest in B (they'll be separated)
#     A_nodes = set([node_order[i] for i in sorted_indices[:size_A]])
#     B_nodes = set([node_order[i] for i in sorted_indices[-size_B:]])
#     return A_nodes, B_nodes
#
#
# def find_min_cut_subsets(
#     G: nx.Graph,
#     ccs: List[Set[Any]],
#     target_A_frac: float = 0.4,
#     target_B_frac: float = 0.1
# ) -> Tuple[Set[Any], Set[Any]]:
#     """
#     Find subsets A and B that minimize cut weight between them.
#
#     Args:
#         G: NetworkX graph
#         ccs: List of connected components (sets of nodes)
#         target_A_frac: Fraction of vertices for set A
#         target_B_frac: Fraction of vertices for set B
#
#     Returns:
#         A: Set of nodes in A
#         B: Set of nodes in B
#         cut_weight: Weight of cut between A and B
#     """
#     n = len(G.nodes())
#     target_A_size = int(n * target_A_frac)
#     target_B_size = int(n * target_B_frac)
#     print(f"Target sizes: ({target_A_size}, {target_B_size})")
#
#     A = set()
#     B = set()
#     remaining_ccs = []
#
#     # Step 1: Greedily assign entire small CCs to A or B
#     for cc in ccs:
#         cc_size = len(cc)
#
#         if len(A) + cc_size <= target_A_size:
#             print("Greedily adding {} samples to A".format(len(cc)))
#             A.update(cc)
#         elif len(B) + cc_size <= target_B_size:
#             print("Greedily adding {} samples to B".format(len(cc)))
#             B.update(cc)
#         else:
#             remaining_ccs.append(cc)
#
#     # Step 2: fill from remaining large CCs as needed.
#     needed_A = target_A_size - len(A)
#     needed_B = target_B_size - len(B)
#     if needed_A <= 0 and needed_B <= 0:
#         pass
#     elif needed_A > 0 and needed_B <= 0:
#         cc = remaining_ccs[0]
#         assert len(cc) > needed_A
#         A.update(sorted(cc)[:needed_A])
#     elif needed_A <= 0 and needed_B > 0:
#         cc = remaining_ccs[0]
#         assert len(cc) > needed_B
#         B.update(sorted(cc)[:needed_B])
#     else:
#         # Need to fill both A and B.
#         if len(remaining_ccs) >= 2:
#             # All remaining ccs are larger than the remaining space.
#             # --> Case 1: Greedily assign chunks from large CCs
#             print("Greedily filling remaining A and B.")
#             cc = remaining_ccs[0]
#             assert len(cc) > needed_A
#             A.update(sorted(cc)[:needed_A])
#
#             cc = remaining_ccs[1]
#             assert len(cc) > needed_B
#             B.update(sorted(cc)[:needed_B])
#         else:
#             # One large component remaining (typical scenario!)
#             # --> Case 2: use spectral clustering.
#             print("Spectral relaxation to fill remaining A and B.")
#             cc_list = list(remaining_ccs[0])
#             subgraph = G.subgraph(cc_list)
#             A_from_cc, B_from_cc = split_component_spectral(
#                 subgraph, needed_A, needed_B
#             )
#             A.update(A_from_cc)
#             B.update(B_from_cc)
#
#     return A, B
#
#
# def train_test_split_mincut_approximation(
#     sample_df: pd.DataFrame,
#     abundance_table_dir: Path,
#     asv_id_subset: Set[str],
#     train_fraction: float,
#     test_fraction: float,
#     distance_matrix: Optional[ASVDistanceMatrix] = None,
# ):
#     """
#     Construct a weighted undirected graph G, where nodes are samples (to be split into test/train), and weights are
#     w_ij = SIMILARITY(sample_i, sample_j).
#
#     SIMILARITY is defined in one of two ways:
#
#     1) If distance_matrix is None, then SIMILARITY is the Jaccard overlap similarity of ASV Identifiers. (not ASV sequences)
#     2) If distance_matrix is given, then SIMILARITY is equal to 1-(DIST_{i,j}/D_MAX), where DIST_{i,j} is the mean distance between all pairs of ASVs in i and j.
#     Namely: DIST_{i,j} = 1/(|i||j|) * \SUM_{asv1 in i} \SUM_{asv2 in j} distance(asv1, asv2)
#
#     The algorithm attempts to find a min-cutset pair amongst all train/test sets of the specified size.
#
#     The algorithm works as follows: First, break G into connected components, and list them out in increasing order of size.
#     For each connected component cc_i, we greedily include it into the training set (or test set, if training is almost full), until |cc_i| is bigger than the remaining capacity.
#
#     To fill the remaining capacity, we do one of the following to fill up the rest of the capacity of train/test.
#
#     1) Greedily separate remaining connected components (if there is more than 1 left) into either training or test. OR
#     2) Perform a Fieldler vector-based sorting of the connected component (if there is only 1 CC left) to find extremal sets.
#
#     This function plots the cut-weights (similarity metric).
#
#     :param sample_df:
#     :param abundance_table_dir:
#     :param asv_id_subset:
#     :param train_fraction:
#     :param test_fraction:
#     :param distance_matrix:
#     :return:
#     """
#     # Collect the list of samples across all projects.
#     proj_ids = list(pd.unique(sample_df['project']))
#     all_samples: List[MicrobiomeSample] = []
#     for proj_id, proj_section in sample_df.groupby('project'):
#         proj = MicrobiomeProject(str(proj_id), abundance_table_dir, sample_df)
#         proj_sample_subset_ids = set(proj_section['srs'])
#         for sample in proj.samples:
#             if sample.sample_id in proj_sample_subset_ids:
#                 n_sample_asvs = len([asv_id for asv_id in sample.asv_ids if asv_id in asv_id_subset])
#                 if n_sample_asvs > 0:
#                     all_samples.append(sample)
#
#     print(f"Splitting {len(all_samples)} samples found in projects: {proj_ids}")
#
#     # Compute weighted adjacency matrix, A[i,j] = # of ASVs shared by sample i and j.
#     n_samples = len(all_samples)
#     n_pairs = int(n_samples * (n_samples - 1) / 2)
#     A = np.zeros((n_samples, n_samples), dtype=float)
#     if distance_matrix is not None:
#         print("Weighted Graph will use weight = (1 - d/d_max) as similarity metric.")
#         def sim_fn(sample1: MicrobiomeSample, sample2: MicrobiomeSample):
#             distance_submat = distance_matrix.submatrix(
#                 [asv_id for asv_id in sample1.asv_ids if asv_id in asv_id_subset],
#                 [asv_id for asv_id in sample2.asv_ids if asv_id in asv_id_subset],
#             )
#             if distance_submat.size == 0:
#                 raise ValueError("Distance submatrix had size 0. Some sample (illegally) had 0 ASVs in filtered collection.")
#             else:
#                 sim_submat = 1 - (distance_submat / distance_matrix.seq_len())  # SIMILARITY = 1 - HAMMING_DIST / SEQLEN
#                 return np.mean(sim_submat)
#     else:
#         print("Weighted Graph will use weight = JACCARD(i,j) as similarity metric.")
#         sim_fn = lambda sample1, sample2: jaccard_similarity(
#             sample1.asv_ids.intersection(asv_id_subset),
#             sample2.asv_ids.intersection(asv_id_subset)
#         )
#
#     for (i, sample_i), (j, sample_j) in tqdm(
#         itertools.combinations(enumerate(all_samples), r=2),
#         total=n_pairs,
#         desc="Sample pair calculation",
#     ):
#         sample_asv_sim = sim_fn(sample_i, sample_j)
#         A[i, j] = sample_asv_sim
#         A[j, i] = sample_asv_sim
#     print("Computed sample similarity matrix of shape {}.".format(A.shape))
#
#     # Create a graph, using "A" as an adjacency matrix. Enumerate all connected components.
#     # first rule out all small/trivial connected components.
#     G = weighted_graph_from_matrix(A, [sample.sample_id for sample in all_samples])
#     ccs = list(nx.connected_components(G))
#     ccs = sorted(ccs, key=lambda x: len(x), reverse=True)
#
#     training_sample_ids, test_sample_ids = find_min_cut_subsets(
#         G,
#         [set(cc) for cc in ccs],
#         target_A_frac=train_fraction,
#         target_B_frac=test_fraction,
#     )
#     cut_weights = cut_edge_weights(G, training_sample_ids, test_sample_ids)
#
#     fig, ax = plt.subplots()
#     ax.hist(cut_weights)
#     ax.set_title("Pairwise similarities: TRAIN <--> TEST")
#     plt.show()
#
#     train_df = sample_df.loc[sample_df['srs'].isin(training_sample_ids)]
#     test_df = sample_df.loc[sample_df['srs'].isin(test_sample_ids)]
#     return train_df, test_df