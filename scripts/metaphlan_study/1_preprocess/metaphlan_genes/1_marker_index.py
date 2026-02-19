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

        print("Max # markers: {}".format(
            max(len(v) for _, v in self.dict_index.items())
        ))

    def preindex(self) -> Dict[str, List[str]]:
        """
        Collect the FASTA ids, and group them by the SGB ID.
        """
        index = defaultdict(list)
        print(f"Reading from index file {self.fai_path.name}")
        with open(self.fai_path, "rt") as fai_file:
            for line in fai_file:
                record_name = line.split("\t")[0]
                if not record_name.startswith("SGB"):
                    continue

                # Each record name is of the form {sgb_id}__{gene_name}, note that gene_name can be duplicated across SGBs, as these are uniprot names.
                sgb_id = record_name.split("__")[0]
                assert sgb_id[-1].isnumeric(), "SGB ID must end with a numeric character. Got: {}".format(sgb_id)
                index[sgb_id].append(record_name)
        return index


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
        marker_fasta_file=Path("/data/bwh-comppath-seq/youn/metaphlan_dset/metaphlan_database") / "markers.fna",
        output_json_path=Path("/data/bwh-comppath-seq/youn/metaphlan_dset/metaphlan_database") / "sgb_marker_index.json.zst",  # should end with ".zst" extension -- will be compressed using zstd.
    )
