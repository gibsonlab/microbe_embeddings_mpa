import torch
from torch.utils.data import Dataset

from ..abundance_profile import AbundanceProfileParser
from .taxa_db import BacterialTaxaDatabase
from .types import *


class OrganismGeneSequenceDataset(Dataset):
    def __init__(
            self,
            db: BacterialTaxaDatabase,
            profile_parser: AbundanceProfileParser
    ):
        """
        Initialize the microbiome dataset.

        :param db: BacterialTaxaDatabase which can translate the taxa IDs found in profile_parser into lists
        of sequences.
        :param profile_parser: A parser that returns the profile(s) to be included.
        """
        super().__init__()
        self.profiles = list(profile_parser.samples())
        self.profile_organisms: List[Sample] = [
            [db.fetch_taxa(taxa_id) for taxa_id in profile.taxa_ids]
            for profile in self.profiles
        ]

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        features = self.profile_organisms[idx]
        labels = torch.from_numpy(self.profiles[idx].abundances_ensure_normalized)
        return features, labels
