from typing import *
from pathlib import Path
import zstandard as zstd
import pandas as pd
import numpy as np

from .asvs import get_asv_sequences


def filter_samples_and_asvs(
        project_metadata: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        abundance_table_dir: Path,
        asv_sequence_file: Path,
        step2_read_count_lb: int = 10,
        step2_num_sample_lb: int = 1,
        step2_min_num_asv: int = 50,
        step2_max_num_asv: int = 250
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, str], int
]:
    project_subset, sample_subset = filter_step1_read_counts(project_metadata, sample_metadata, abundance_table_dir)
    asv_id_subset, sample_max_num_asvs, sample_id_subset_post_filter = filter_step2_select_subset_asvs(
        project_subset['project'].to_list(),
        set(sample_subset['srs'].tolist()),
        abundance_table_dir,
        read_count_lb=step2_read_count_lb,
        num_sample_lb=step2_num_sample_lb,
        min_num_asv=step2_min_num_asv,
        max_num_asv=step2_max_num_asv,
    )

    print("Num. project-relevant ASVs:", len(asv_id_subset))
    print("Per-sample max num. ASV:", sample_max_num_asvs)

    # ======= Update the sample_subset and project_subset dataframes, since we removed some samples with 0 filtered ASVs.
    sample_subset = sample_subset.loc[sample_subset['srs'].isin(sample_id_subset_post_filter)]
    project_subset = project_subset.loc[project_subset['project'].isin(set(sample_subset['project']))]

    # ======= Fetch the ASV sequences.
    asv_seqs_subset = {
        asv_id: asv_seq
        for asv_id, asv_seq in get_asv_sequences(asv_sequence_file)
        if asv_id in asv_id_subset
    }

    assert len(asv_seqs_subset) == len(asv_id_subset)
    return project_subset, sample_subset, asv_seqs_subset, sample_max_num_asvs


def filter_step1_read_counts(
        project_metadata: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        abundance_table_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    First, filter by project ID (American Gut, PRJEB11419) and restrict to US/CA subjects and Illumina MiSeq experiments.
    Then, filter out those samples which have an abnormal amount of read counts (outside of 2.5% <--> 97.5% quantile)

    :param project_metadata:
    :param sample_metadata:
    :param abundance_table_dir: The directory containing the HMC sample information.
    """
    project_subset = project_metadata
    sample_subset = sample_metadata
    
    # ========== attach read depth (total # of ASV reads) as a column.
    sample_total_read_counts_all_list: List[pd.Series] = []
    for proj_id, proj_section in sample_subset.groupby('project'):
        with zstd.open(abundance_table_dir / f"{proj_id}.txt.zst", "rt") as f:
            counts_table = pd.read_csv(f, sep='\t').set_index("asv")  # load from file
            counts_table = counts_table[list(proj_section['srs'])]  # restrict to samples from sample_subset
            sample_total_reads = counts_table.sum(axis=0)  # compute total ASV read counts in each sample
            sample_total_read_counts_all_list.append(sample_total_reads)  # append to list for concatenation.

    sample_total_read_counts_all: pd.Series = pd.concat(sample_total_read_counts_all_list).rename("read_counts").astype(int)
    assert sample_total_read_counts_all.shape[0] == sample_subset.shape[
        0], f"Expected read count for each sample. Got: {sample_total_read_counts_all.shape[0]} vs {sample_subset.shape[0]}"

    sample_subset = sample_subset.merge(
        sample_total_read_counts_all.to_frame(),
        left_on=['srs'], right_index=True,
        how='left'
    )

    # ========== filter by read count threshold. (perform this on a per-project basis)
    sections = []
    for proj_id, sample_project_section in sample_subset.groupby("project"):
        print(f"[Project {proj_id}] Read count statistics:")
        print(sample_project_section['read_counts'].describe().to_frame())

        read_count_ub = np.quantile(sample_project_section['read_counts'], q=0.950)
        read_count_lb = np.quantile(sample_project_section['read_counts'], q=0.050)
        print(f"[Project {proj_id}] Using read count threshold of {read_count_lb} < x < {read_count_ub}")
        sample_project_section = sample_project_section.loc[
            (sample_project_section['read_counts'] <= read_count_ub) & (
                        sample_project_section['read_counts'] >= read_count_lb)
            ]
        sections.append(sample_project_section)
        print("*********************************")

    sample_subset = pd.concat(sections, ignore_index=True)
    print("[*] Stage 2 filter: {} samples remaining".format(sample_subset.shape[0]))
    del sections

    print(sample_subset.groupby('project')['iso'].value_counts().rename("count").to_frame())

    # ========== Now cross-reference back after filtering by region/read depth.
    project_subset = project_subset.loc[project_subset['project'].isin(set(sample_subset['project']))]
    return project_subset, sample_subset



"""
Next, restrict the set of ASVs to only those found in the filtered set of samples.
Apply an additional filter on the ASVs: only keep ASVs which appear in >=1 sample with a read count of >=10.
Finally, remove samples with 0 of these filtered ASVs.
"""
def filter_step2_select_subset_asvs(
        project_ids: List[str],
        sample_subset: Set[str],
        abundance_table_dir: Path,
        read_count_lb: int,
        num_sample_lb: int,
        min_num_asv: int,
        max_num_asv: int,
) -> Tuple[Set[str], int, Set[str]]:
    """
    :param read_count_lb: Consider ASVs exceeding this count in `num_sample_lb` samples.
    :param num_sample_lb: Consider ASVs exceeding `read_count_lb` count in this many samples.

    :param min_num_asv: After the ASV screening, only keep samples exceeding this number of ASVs.
    :param max_num_asv: After the ASV screening, only keep samples below this number of ASVs.
    
    :return: A triple of objects (Set of ASV IDs, Max # of ASV per sample, subset of sample IDs). 
    The subset of sample IDs is guaranteed to be a subset of the input 'sample_subset'; it should be used for further downstream pre-filtering of samples.
    """
    asv_set = set()
    max_sample_asv = 0
    new_sample_id_subset = set()

    # Load all asv count tables.
    project_asv_tables = dict()
    for project_id in project_ids:
        with zstd.open(abundance_table_dir / f"{project_id}.txt.zst", "rt") as f:
            table = pd.read_csv(f, sep='\t').set_index("asv")
            project_sample_ids = set(table.columns)
            
            table = table[list(sample_subset.intersection(project_sample_ids))]
            project_asv_tables[project_id] = table
            print("[* Pre-ASV selection filtering] PROJECT {}: {} samples, {} ASVs".format(project_id, len(table.columns), len(table.index)))

    """
    Filter stage 1:

    For each ASV, which samples have this ASV with count >= read_count_lb? Then count the # of samples (np.sum) and filter by num_sample_lb.
    Note: compute the total sample tally across all projects (call df.add)
    """
    print(f"Filtering ASVs by (# samples with count >= {read_count_lb}) >= {num_sample_lb}")
    asv_n_samples_overall = None
    for project_id, project_table in project_asv_tables.items():
        asv_n_samples = np.sum(project_table >= read_count_lb, axis=1)
        if asv_n_samples_overall is None:
            asv_n_samples_overall = asv_n_samples
        else:
            asv_n_samples_overall = asv_n_samples_overall.add(asv_n_samples, fill_value=0).astype(int)

    # Restrict table to those ASVs where # samples >= num_sample_lb
    for project_id, project_table in project_asv_tables.items():
        asv_mask = asv_n_samples_overall >= num_sample_lb
        # not all ASVs in asv_mask are present in project_table.
        project_table = project_table[asv_mask.loc[project_table.index]]
        assert set(asv_mask.loc[project_table.index].index) == set(project_table.index), "Pandas syntax didn't select the correct rows indexed by ASV id."
        project_asv_tables[project_id] = project_table

    """
    Filter stage 2:

    For each Sample, only keep it if it has 'min_num_asv' filtered ASVs from stage 1.
    Also, for memory constraints during training, filter by 'max_num_asv'.
    """
    print(f"Filtering Samples by ASV count between {min_num_asv} and {max_num_asv}")
    for project_id, project_table in project_asv_tables.items():
        num_asv_per_sample = np.sum(project_table > 0, axis=0)
        samples_with_lb_asvs = num_asv_per_sample.loc[
            (num_asv_per_sample >= min_num_asv) & (num_asv_per_sample <= max_num_asv)
        ]
        project_table = project_table.loc[:, samples_with_lb_asvs.index.to_list()]
        project_asv_tables[project_id] = project_table


    """
    Finalize and report summary.
    """
    for project_id, project_table in project_asv_tables.items():
        print("[* Post-ASV selection filtering] PROJECT {}: {} samples, {} ASVs".format(project_id, len(project_table.columns), len(project_table.index)))
        for sample_id in project_table.columns.to_list():
            new_sample_id_subset.add(sample_id)
            sample_n_asvs = np.sum(project_table[sample_id] > 0)
            max_sample_asv = max(max_sample_asv, sample_n_asvs)

        this_asv_ids = set(project_table.index.to_list())
        print("This project contributes {} new ASVs after filtering.".format(len(this_asv_ids.difference(asv_set))))
        asv_set = asv_set.union(this_asv_ids)
    return asv_set, max_sample_asv, new_sample_id_subset 
