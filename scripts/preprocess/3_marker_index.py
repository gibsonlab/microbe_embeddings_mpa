"""
Take the fasta files output in step 1, and compile the information into a database.
The database is a dictionary, mapping SGB IDs into collections of markers.
"""
from typing import *
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd
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
        Collect the FASTA ids, and group them by the source genome ID.
        """
        index = defaultdict(list)
        with open(self.fai_path, "rt") as fai_file:
            for line in fai_file:
                record_name = line.split("\t")[0]
                genome_id = record_name.split("__")[-1]
                index[genome_id].append(record_name)
        return index

    def fetch_marker_names(self, centroid_genome_id: str) -> List[str]:
        """
        Answers the query for input genome ID by returning the list of constituent marker fasta record IDs.
        """
        if centroid_genome_id not in self.dict_index:
            raise KeyError(f"Genome ID {centroid_genome_id} not found in index.")
        return self.dict_index[centroid_genome_id]


def main(
        metadata_csv_path: Path,
        marker_fasta_file: Path,
        output_json_path: Path,
):
    phylo_marker_index = MarkerIndex(marker_fasta_file)
    sgb_dict = dict()
    sgb_metadata = pd.read_csv(metadata_csv_path, sep="\t")
    for _, row in sgb_metadata.iterrows():
        sgb_prefix = row['Label']
        sgb_number = row['ID']

        sgb_id = f"{sgb_prefix}{sgb_number}"  # as found in MetaPhlAn4 database, e.g. "SGB1092"
        centroid_genome_id = row['SGB centroid']  # these are the IDs found in the PhyloPhlAn fasta marker files.
        sgb_dict[sgb_id] = {
            'centroid': centroid_genome_id,
            'markers': phylo_marker_index.fetch_marker_names(centroid_genome_id)
        }

    # Save the SGB dictionary to file.
    output_json_path.parent.mkdir(exist_ok=True)
    with zstd.open(output_json_path, "wt") as json_file:
        json.dump(sgb_dict, json_file, indent=4)


if __name__ == "__main__":
    main(
        metadata_csv_path=Path("todo_dir") / "sgb_metadata.csv",
        marker_fasta_file=Path("todo_dir") / "all_markers.fna.bgz",
        output_json_path=Path("todo_dir") / "sgb_marker_index.json.zst",  # should end with ".zst" extension -- will be compressed using zstd.
    )
