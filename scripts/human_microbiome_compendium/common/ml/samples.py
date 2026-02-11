from typing import *
from dataclasses import dataclass
from pathlib import Path
import zstandard as zstd
import pandas as pd
import numpy as np


@dataclass
class SampleMetadata:
    library_strategy: str
    library_source: str
    total_bases: int
    instrument: str
    geo_loc_name: str
    iso: str
    region: str


class MicrobiomeSample:
    def __init__(self, sample_id: str, read_counts: Dict[str, float], sample_metadata_table: pd.DataFrame):
        self.sample_id = sample_id
        # self.read_counts = read_counts
        self.metadata = self.get_metadata(sample_metadata_table)
        self.asv_ordering = sorted(read_counts.keys())
        self.asv_id_set = set(self.asv_ordering)
        self.asv_reads_full = np.array([read_counts[asv_id] for asv_id in self.asv_ordering])
        self.asv_indices = {asv_id: i for i, asv_id in enumerate(self.asv_ordering)}

    @property
    def asv_ids(self) -> Set[str]:
        return self.asv_id_set

    # def set_count(self, asv_id: str, count_value: int) -> None:
    #     self.read_counts[asv_id] = count_value

    def get_metadata(self, sample_metadata_table: pd.DataFrame) -> SampleMetadata:
        df_row = sample_metadata_table.loc[sample_metadata_table['srs'] == self.sample_id]
        assert df_row.shape[0] == 1, "Expected 1 row of metadata for id {}, got: {}".format(self.sample_id, df_row.shape[0])
        df_row = df_row.head(1)
        return SampleMetadata(
            df_row['library_strategy'], df_row['library_source'], df_row['total_bases'],
            df_row['instrument'], df_row['geo_loc_name'], df_row['iso'], df_row['region']
        )

    # def read_count_array(self, asv_id_subset: Optional[Set[str]] = None) -> Tuple[List[str], np.ndarray]:
    #     """
    #     Return read counts, formatted as a numpy array.
    #     Note that the ordering of the read counts is not guaranteed, and may depend on python version! Check the List[str] part of this method's output for the resulting order.
    #     :return: A pair (asv_ids, read_counts), where `read_counts` is the numpy array, and `asv_ids` is the order of the ASVs indexing the array.
    #     """
    #     if asv_id_subset is None:
    #         target_asv_ids = self.asv_ids
    #     else:
    #         target_asv_ids = asv_id_subset.intersection(self.asv_ids)

    #     if len(target_asv_ids) == 0:
    #         raise ValueError(f"Sample ID = {self.sample_id}, No ASVs in the restricted set!")

    #     asv_ids = []
    #     asv_reads = []
    #     for asv_id in target_asv_ids:
    #         asv_read_count = self.read_counts[asv_id]
    #         asv_ids.append(asv_id)
    #         asv_reads.append(asv_read_count)
    #     return asv_ids, np.array(asv_reads, dtype=int)

    def relative_abundance_array(self, asv_id_subset: Optional[Set[str]] = None) -> Tuple[List[str], np.ndarray]:
        """
        Return estimated relative abundances, formatted as a numpy array.
        :return: A pair (asv_ids, rel_abunds), where `rel_abunds` is the numpy array, and `asv_ids` is the order of the ASVs indexing the array.
        """
        if asv_id_subset is None:
            return self.asv_ordering, self.asv_reads_full / np.sum(self.asv_reads_full)

        subset_order = list(asv_id_subset.intersection(self.asv_id_set))
        subset_indices = [self.asv_indices[_aid] for _aid in subset_order]
        counts_subset = self.asv_reads_full[subset_indices]
        return subset_order, counts_subset / np.sum(counts_subset)
        # asv_ids, read_counts = self.read_count_array(asv_id_subset)
        # return asv_ids, read_counts / np.sum(read_counts)

    def num_asvs(self) -> int:
        return len(self.read_counts)

    def __str__(self) -> str:
        return self.sample_id

    def __repr__(self) -> str:
        return f"Sample[{self.sample_id}]"


class MicrobiomeProject:
    def __init__(
            self,
            project_id: str,
            abundance_table_dir: Path,
            sample_metadata_table: pd.DataFrame
    ):
        self.project_id = project_id
        self.samples = self.all_samples(project_id, abundance_table_dir, sample_metadata_table)

    def __str__(self) -> str:
        return "Project[{}]".format(self.project_id)

    def __repr__(self) -> str:
        return "Project[{}]".format(self.project_id)

    @staticmethod
    def all_samples(
            project_id: str,
            abundance_table_dir: Path,
            sample_subset_table: pd.DataFrame
    ) -> List[MicrobiomeSample]:
        samples: Dict[str, Dict[str, float]] = dict()

        # Create the Abundance profile objects.
        sample_subset_table_in_proj = sample_subset_table.loc[
            sample_subset_table['project'] == project_id
        ]
        sample_id_subset = [str(srs_id) for srs_id in sample_subset_table_in_proj['srs']]
        for sample_id in sample_id_subset:
            assert sample_id.startswith("DRS") or sample_id.startswith("SRS") or sample_id.startswith(
                "ERS"), f"In {project_id}, expected sample ID to start with `DRS`, `SRS` or `ERS`. Got: {sample_id}"

            # samples[sample_id] = MicrobiomeSample(sample_id, sample_subset_table)
            samples[sample_id] = dict()

        # Parse the large abundance table. (this contains more samples than we want)
        with zstd.open(abundance_table_dir / f"{project_id}.txt.zst", "rt") as abund_file:
            header_line = abund_file.readline()
            assert header_line.startswith("asv\t"), f"Unrecognized format for abundance table: {project_id}"

            abund_sample_order = header_line.strip().split("\t")[1:]
            assert set(samples.keys()).issubset(set(abund_sample_order)), "Sample Subset IDs must be a subset of Header line of ASV counts table."
            for row in abund_file:
                tokens = row.strip().split("\t")
                asv_id = tokens[0]
                assert len(abund_sample_order) == len(tokens) - 1, "Mismatch in the number of abundance tokens!"
                for sample_id, abund_token in zip(abund_sample_order, tokens[1:]):
                    if sample_id in samples:
                        sample_dict = samples[sample_id]
                        asv_count = int(float(abund_token))
                        if asv_count > 0:
                            sample_dict[asv_id] = asv_count
                            # sample_obj.set_count(asv_id, asv_count)

        return [MicrobiomeSample(s_id, samples[s_id], sample_subset_table) for s_id in sample_id_subset]