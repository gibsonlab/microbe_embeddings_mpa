from abc import ABC, abstractmethod
from typing import Dict
from pathlib import Path
import zstandard as zstd
import json
from pyfaidx import Fasta

from .types import *


class BacterialTaxaDatabase(ABC):
    @abstractmethod
    def fetch_taxa(self, taxa_id: str) -> BacterialTaxa:
        pass


""" Implementations below """

class MetaphlanTaxaDatabase(BacterialTaxaDatabase):
    def __init__(self, json_index_path: Path, fasta_path: Path):
        """
        Requires a ZST-compressed JSON file, representing a dictionary mapping SGB ids to a list of FASTA record IDs.
        The accompanying FASTA file must contain an entry with this record ID, so that the entry is a nucleotide/amino
        acid sequence that is a marker gene for the source SGB id.

        :param json_index_path: A path to the ZST-compressed JSON index file.
        :param fasta_path: A path to a pre-existing FASTA file containing the raw marker sequences.
        """
        print("json_index_path: ", json_index_path)
        with zstd.open(json_index_path, "rt") as f:
            sgb_marker_index = json.load(f)

        fasta = Fasta(fasta_path)
        self.catalogue: Dict[str, BacterialTaxa] = {}

        print(f"Loading marker sequence catalog from {json_index_path}")
        for sgb_id_numeric_str, seq_record_ids in sgb_marker_index.items():
            # remember to attach the "SGB" prefix!
            sgb_id = f'SGB{sgb_id_numeric_str}'
            self.catalogue[sgb_id] = [str(fasta[seq_id]) for seq_id in seq_record_ids]
        print("Loaded {} SGBs.".format(len(self.catalogue)))

    def fetch_taxa(self, taxa_id: str) -> BacterialTaxa:
        return self.catalogue[taxa_id]
