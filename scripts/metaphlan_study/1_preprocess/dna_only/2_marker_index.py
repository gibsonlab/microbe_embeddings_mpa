"""
Take the fasta files output in step 1, and compile the information into a database.
The database is a dictionary, mapping SGB IDs into collections of markers.
"""
from typing import *
from pathlib import Path
from collections import defaultdict
import json
import zstandard as zstd
from pyfaidx import Fasta


class MarkerIndex:
    def __init__(self, marker_fasta_file):
        self.marker_fasta_file = Fasta(marker_fasta_file)  # this also indexes the file (.fai file)
        self.fai_path = Path(marker_fasta_file.parent / f'{marker_fasta_file.name}.fai')
        assert self.fai_path.exists(), "pyfaidx did not index the fasta file as expected."
        self.dict_index = self.preindex()

    def preindex(self) -> Dict[str, List[str]]:
        """
        Collect the FASTA ids, and group them by the SGB ID.
        """
        index = defaultdict(list)
        with open(self.fai_path, "rt") as fai_file:
            for line in fai_file:
                record_name = line.split("\t")[0]

                # Each record name is of the form {gene_name}:{sgb_id}__{genome_acc}
                gene_and_sgb = record_name.split("__")[0]
                tokens = gene_and_sgb.split(":")
                assert len(tokens) == 2, f"Unexpected parse of ID: {record_name}"
                gene_id, sgb_id = tokens
                index[sgb_id].append(record_name)
        return index

    def fetch_marker_names(self, sgb_id: str) -> List[str]:
        """
        Answers the query for input genome ID by returning the list of constituent marker fasta record IDs.
        """
        if sgb_id not in self.dict_index:
            raise KeyError(f"SGB id {sgb_id} not found in index.")
        return self.dict_index[sgb_id]

    def items(self) -> Iterator[Tuple[str, List[str]]]:
        yield from self.dict_index.items()


def main(
        marker_fasta_file: Path,
        output_json_path: Path,
):
    assert marker_fasta_file.exists(), f"Marker FASTA file {marker_fasta_file} does not exist!"
    phylo_marker_index = MarkerIndex(marker_fasta_file)

    # Save the SGB dictionary to file.
    output_json_path.parent.mkdir(exist_ok=True)
    with zstd.open(output_json_path, "wt") as json_file:
        json.dump(phylo_marker_index.dict_index, json_file, indent=4)


if __name__ == "__main__":
    main(
        marker_fasta_file=Path("/data/bwh-comppath-seq/youn/metaphlan_dset/phylophlan_data/processed/dna_only") / "markers.fna",
        output_json_path=Path("/data/bwh-comppath-seq/youn/metaphlan_dset/phylophlan_data/processed/dna_only") / "sgb_marker_index.json.zst",  # should end with ".zst" extension -- will be compressed using zstd.
    )
