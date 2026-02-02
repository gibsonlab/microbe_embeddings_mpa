from typing import *
from pathlib import Path
import zstandard as zstd
import pandas as pd
import numpy as np

from .asvs import get_asv_sequences


def filter_samples_and_asvs(
        project_metadata: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        target_project_ids: Set[str],
        abundance_table_dir: Path,
        asv_sequence_file: Path,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, str], int
]:
    project_subset, sample_subset = filter_step1_project(project_metadata, sample_metadata, target_project_ids, abundance_table_dir)
    asv_id_subset, sample_max_num_asvs, sample_id_subset_post_filter = filter_step2_select_subset_asvs(
        project_subset['project'].to_list(),
        set(sample_subset['srs'].tolist()),
        abundance_table_dir,
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


def filter_step1_project(
        project_metadata: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        target_project_ids: Set[str],
        abundance_table_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    First, filter by project ID (American Gut, PRJEB11419) and restrict to US/CA subjects and Illumina MiSeq experiments.
    Then, filter out those samples which have an abnormal amount of read counts (outside of 2.5% <--> 97.5% quantile)

    :param project_metadata:
    :param sample_metadata:
    :param target_project_ids:
    :param abundance_table_dir: The directory containing the HMC sample information.
    """
    project_subset = project_metadata.loc[
        (project_metadata['condition'] == 'healthy')
        & (project_metadata['amplicon'] == 'v4')
        & project_metadata['project'].isin(target_project_ids),
        :
    ]

    sample_subset = sample_metadata.loc[
        sample_metadata['iso'].isin({"US", "CA"})  # filter by region.
        & sample_metadata['project'].isin(set(project_subset['project']))  # filter by results of "project_subset".
        & (sample_metadata['instrument'] == 'Illumina MiSeq')  # filter by sequencing instrument.
        ]
    print("[*] Stage 1 filter: {} samples remaining".format(sample_subset.shape[0]))

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

        read_count_ub = np.quantile(sample_project_section['read_counts'], q=0.975)
        read_count_lb = np.quantile(sample_project_section['read_counts'], q=0.025)
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
) -> Tuple[Set[str], int, Set[str]]:
    """
    :return: A triple of objects (Set of ASV IDs, Max # of ASV per sample, subset of sample IDs).
    The subset of sample IDs is guaranteed to be a subset of the input 'sample_subset'; it should be used for further downstream pre-filtering of samples.
    """
    asv_set = set()
    max_sample_asv = 0
    new_sample_id_subset = set()
    for project_id in project_ids:
        with zstd.open(abundance_table_dir / f"{project_id}.txt.zst", "rt") as f:
            table = pd.read_csv(f, sep='\t').set_index("asv")
            table = table[list(sample_subset)]
            print(
                "[* Pre-ASV selection filtering] PROJECT {}: {} samples, {} ASVs".format(project_id, len(table.columns),
                                                                                         len(table.index)))

            # # ============ debug
            # read_depths = table.sum(axis=0)
            # fig, ax = plt.subplots(1, 1)
            # sb.histplot(read_depths, ax=ax)

            # _counts = table.to_numpy()
            # _counts_sum_across_samples = np.sum(_counts, axis=1)  # PER ASV: total counts across samples, dimension of array = (# asvs)
            # print("# asvs with total reads > 5 across samples: {}".format(
            #     np.sum(_counts_sum_across_samples > 5)
            # ))
            # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            # for sample_filter in [1, 2, 3, 4, 5]:  ## each curve
            #     _asv_counts = []
            #     count_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            #     for ct_filter in count_ticks:  # x-axis
            #         _thresholded_presence_per_asv_per_sample = (_counts >= ct_filter)  # table [ASV x SAMPLE]
            #         _counts_thresholded_samples = np.sum(_thresholded_presence_per_asv_per_sample, axis=1)  # PER ASV: how many samples with count > 2, dimension of array = (# asvs)
            #         # Example: _counts_thresholded_samples = [ASV0: 1 sample, ASV1: 5 sample, ASV2: 10 sample, ...]
            #         y_to_plot = np.sum(_counts_thresholded_samples >= sample_filter)  # how many ASVS meeting criteria: at least `sample_filter` samples with count > 2
            #         _asv_counts.append(int(y_to_plot))
            #     ax.plot(count_ticks, _asv_counts, label='sample filter = {}'.format(sample_filter))
            #         # print("# asvs present (present means > 2 reads in sample) in more than 1 samples: {}".format(
            #         #     np.sum(_counts_thresholded_samples >= 1)
            #         # ))
            # ax.set_xticks(count_ticks)
            # ax.set_xlabel('Count filter')
            # ax.set_yscale('log')
            # plt.legend()
            # # ============ end debug

            read_count_lb = 10
            num_sample_lb = 1

            print(f"Filtering ASVs by (# samples with count >= {read_count_lb}) >= {num_sample_lb}")
            asv_n_samples = np.sum(table >= read_count_lb,
                                   axis=1)  # per asv: which samples have this ASV with count >= 3? Then count the # of samples (np.sum)
            table = table.loc[
                asv_n_samples >= num_sample_lb, :]  # restrict table to those ASVs where # samples (satisfying above threshold) >= 3

            min_num_asv = 50
            print(f"Filtering samples by (# filtered asv) >= {min_num_asv}")
            num_asv_per_sample = np.sum(table > 0, axis=0)
            samples_with_lb_asvs = num_asv_per_sample.loc[num_asv_per_sample >= min_num_asv]
            table = table.loc[:, samples_with_lb_asvs.index.to_list()]

            # ======= DEBUG
            # display(num_asv_per_sample.sort_values())
            # fig, ax = plt.subplots(1, 1)
            # ax.hist(num_asv_per_sample, bins=20)
            # ax.set_ylabel('# of samples')
            # ax.set_xlabel('# of ASVs')
            # ======= END DEBUG

            print("[* Post-ASV selection filtering] PROJECT {}: {} samples, {} ASVs".format(project_id,
                                                                                            len(table.columns),
                                                                                            len(table.index)))
            for sample_id in table.columns.to_list():
                new_sample_id_subset.add(sample_id)
                sample_n_asvs = np.sum(table[sample_id] > 0)
                max_sample_asv = max(max_sample_asv, sample_n_asvs)

            for asv_id in table.index.to_list():
                asv_set.add(asv_id)
    return asv_set, max_sample_asv, new_sample_id_subset
