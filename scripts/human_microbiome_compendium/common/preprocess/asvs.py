from typing import Iterator, Tuple, Dict
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
    input_subset = set(sequences_dict.keys())

    # this is always a subset of input_subset. may or may not be the entire set, depending on what `sample_df` gets passed in.
    output_subset = set()

    for proj_id in sample_df.groupby("project"):
        # load the sample table dir.
        with zstd.open(abundance_table_dir / f"{proj_id}.txt.zst", "rt") as f:
            proj_asvs = set(pd.read_csv(f, sep='\t')['asv']).intersection(input_subset)
            output_subset.update(proj_asvs)

    # Now write these ASVs to file.
    with open(out_path, "wt") as f:
        for asv_id in sorted(output_subset):
            print(asv_id, file=f)
