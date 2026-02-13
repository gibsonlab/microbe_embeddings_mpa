from typing import Iterator, Tuple, Dict, Set
from pathlib import Path
import zstandard as zstd
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def get_asv_sequences(asv_sequence_file: Path) -> Iterator[Tuple[str, str]]:
    """
    Fetch all of the ASV sequences from the asv_sequence_file location.
    :return: A generator over (<ASV_ID>, <ASV_SEQ>) tuples.
    """
    with zstd.open(asv_sequence_file, "rt") as f:
        header_line = f.readline()
        assert header_line.startswith("#OTU"), "Expected file to start with the header: `#OTU`"

        for line in f:
            tokens = line.strip().split("\t")
            assert len(tokens) == 2, f"Expected exactly two tokens. Got: {line}"

            asv_id, asv_seq = tokens[0], tokens[1]
            assert asv_id.startswith("ASV"), f"Expected token #1 to have prefix `ASV`. Got: {asv_id}"
            yield asv_id, asv_seq


def dict_to_fasta(sequences_dict: Dict[str, str], output_file: Path) -> None:
    """
    Write sequences from dictionary to multi-FASTA file using BioPython.

    :param sequences_dict: Dictionary mapping sequence IDs to sequences
    :param output_file: Path to output FASTA file
    """
    seq_records: list[SeqRecord] = []
    for seq_id, sequence in sequences_dict.items():
        seq_record: SeqRecord = SeqRecord(
            Seq(sequence),
            id=seq_id,
            description=""
        )
        seq_records.append(seq_record)

    with open(output_file, "w") as handle:
        SeqIO.write(seq_records, handle, "fasta")

    print(f"Wrote {len(seq_records)} sequences to {output_file}")


def get_asvs_in_project(proj_id: str, abundance_table_dir: Path, sample_subset: Set[str], filt_asv_ids_subset: Set[str]) -> Set[str]:
    # load the sample table dir.
    with zstd.open(abundance_table_dir / f"{proj_id}.txt.zst", "rt") as f:
        table = pd.read_csv(f, sep='\t').set_index("asv")

        # restrict to samples which appear in the input.
        project_sample_ids = set(table.columns)
        table = table[list(sample_subset.intersection(project_sample_ids))]

        # restrict to those ASVs which appear in the specified subset.
        existing_asv_idxs = filt_asv_ids_subset.intersection(table.index)
        table = table.loc[list(existing_asv_idxs)]

        # delete all ASVs which don't appear in any sample.
        table = table[(table != 0.0).any(axis=1)]

        proj_asvs = set(str(s) for s in table.index)
        return proj_asvs


def dump_asv_ids(sequences_dict: Dict[str, str], sample_df: pd.DataFrame, abundance_table_dir: Path, out_path: Path) -> None:
    """
    Dump a text file of ASV ids to the target file, one line per id.
    Only takes ASV ids which appear in sequences_dict, AND which appear in samples contained in sample_df.

    :param sequences_dict: Dictionary mapping sequence IDs to sequences
    :param sample_df: Sample dataframe specifying the sample metadata rows that you wish to keep ASV IDs from.
    If wanting to dump ASV ids per project, this dataframe should be sliced beforehand.
    :param abundance_table_dir: Abundance table directory
    :param out_path: A path to an output file.
    """
    sample_subset = set(sample_df['srs'])
    input_asv_subset = set(sequences_dict.keys())

    # this is always a subset of input_subset. may or may not be the entire set, depending on what `sample_df` gets passed in.
    output_subset = set()

    sample_proj_ids = set(sample_df['project'])
    for proj_id in sample_proj_ids:
        proj_asvs = get_asvs_in_project(proj_id, abundance_table_dir, sample_subset, input_asv_subset)
        output_subset.update(proj_asvs)

    # Now write these ASVs to file.
    with open(out_path, "wt") as f:
        for asv_id in sorted(output_subset):
            print(asv_id, file=f)
