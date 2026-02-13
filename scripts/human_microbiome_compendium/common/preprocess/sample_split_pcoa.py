""" Option 3: Train samples should not share any ASVs with test samples. """
from typing import *
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

from ..ml import *
from ..util import ASVDistanceMatrix
from scipy.special import kl_div


def jaccard_similarity(x: Set, y: Set) -> float:
    numer = len(x.intersection(y))
    denom = len(x.union(y))
    return numer / denom


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return kl_div(p, q).sum()


def jensen_shannon_between_samples(x: MicrobiomeSample, y: MicrobiomeSample, asv_id_subset: Set[str]) -> float:
    x_ids = x.asv_ids.intersection(asv_id_subset)
    y_ids = y.asv_ids.intersection(asv_id_subset)
    all_ids = x_ids.union(y_ids)
    p = x.relative_abundance_array_padded(asv_id_order=all_ids)
    q = y.relative_abundance_array_padded(asv_id_order=all_ids)
    m = 0.5 * (p + q)
    return 0.5 * (kl_div(p, m) + kl_div(q, m))


def compute_sample_distance_matrix(
        sample_df: pd.DataFrame,
        abundance_table_dir: Path,
        asv_id_subset: Set[str],
        distance_method: str = 'jaccard',
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
    sample_id_order = [s.sample_id for s in all_samples]
    print(f"Computing distance matrix between {len(all_samples)} samples found in projects: {proj_ids}")

    # Compute weighted adjacency matrix, A[i,j] = # of ASVs shared by sample i and j.
    n_samples = len(all_samples)
    n_pairs = int(n_samples * (n_samples - 1) / 2)
    A = np.zeros((n_samples, n_samples), dtype=float)
    if distance_method == 'precomputed':
        if asv_distance_matrix is not None:
            print("Weighted Graph will use weight = (1 - d/d_max) as similarity metric.")

            def dist_fn(sample1: MicrobiomeSample, sample2: MicrobiomeSample):
                distance_submat = asv_distance_matrix.submatrix(
                    [asv_id for asv_id in sample1.asv_ids if asv_id in asv_id_subset],
                    [asv_id for asv_id in sample2.asv_ids if asv_id in asv_id_subset],
                )
                if distance_submat.size == 0:
                    raise ValueError(
                        "Distance submatrix had size 0. Some sample (illegally) had 0 ASVs in filtered collection.")
                else:
                    return np.mean(distance_submat)
        else:
            raise ValueError("If distance_method is `precomputed`, asv_distance_matrix must be provided.")
    elif distance_method == 'jaccard':
        print("Using DISTANCE = 1 - JACCARD(i,j) as distance metric.")
        dist_fn = lambda sample1, sample2: 1 - jaccard_similarity(
            sample1.asv_ids.intersection(asv_id_subset),
            sample2.asv_ids.intersection(asv_id_subset)
        )
    elif distance_method == 'jensen_shannon':
        print("Using DISTANCE = JensenShannon(i,j) = 0.5 (KL(i, m) + KL(j, m)) as a metric.")
        dist_fn = lambda sample1, sample2: jensen_shannon_between_samples(sample1, sample2, asv_id_subset)
    else:
        raise ValueError(f"Unsupported distance method `{distance_method}`.")

    for (i, sample_i), (j, sample_j) in tqdm(
            itertools.combinations(enumerate(all_samples), r=2),
            total=n_pairs,
            desc="Sample pair calculation",
    ):
        sample_asv_dist = dist_fn(sample_i, sample_j)
        A[i, j] = sample_asv_dist
        A[j, i] = sample_asv_dist
    print("Computed sample similarity matrix of shape {}.".format(A.shape))
    return A, sample_id_order