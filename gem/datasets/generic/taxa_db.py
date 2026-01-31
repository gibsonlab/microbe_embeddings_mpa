from abc import ABC, abstractmethod
from typing import Dict
from pathlib import Path
import zstandard as zstd
import json
from pyfaidx import Fasta
from tqdm import tqdm

from .types import *


class BacterialTaxaDatabase(ABC):
    @abstractmethod
    def fetch_taxa(self, taxa_id: str) -> BacterialTaxa:
        pass

    @abstractmethod
    def __contains__(self, taxa_id: str) -> bool:
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
        try:
            with zstd.open(json_index_path, "rt") as f:
                sgb_marker_index = json.load(f)
        except Exception as e:
            print("Fatal error while reading sequence index file {json_index_path}: {cause}".format(
                json_index_path=json_index_path,
                cause=str(e)
            ))


        fasta = Fasta(fasta_path)
        self.catalogue: Dict[str, BacterialTaxa] = {}

        print(f"Loading marker sequence catalog from {json_index_path}")
        for sgb_id_numeric_str, seq_record_ids in tqdm(sgb_marker_index.items(), desc='Marker-DB', unit=' SGB'):
            # remember to attach the "SGB" prefix!
            sgb_id = f'SGB{sgb_id_numeric_str}'
            self.catalogue[sgb_id] = [str(fasta[seq_id]) for seq_id in seq_record_ids]
        print("Loaded {} SGBs.".format(len(self.catalogue)))

    def fetch_taxa(self, taxa_id: str) -> BacterialTaxa:
        return self.catalogue[taxa_id]

    def __contains__(self, taxa_id: str) -> bool:
        return taxa_id in self.catalogue
