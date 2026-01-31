from abc import abstractmethod, ABC
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset

from gem.datasets.mpa import MetaphlanProfileCollection

Gene = str
BacterialTaxa = List[Gene]
Sample = List[BacterialTaxa]


class BacterialTaxaDatabase(ABC):
    @abstractmethod
    def fetch_taxa(self, taxa_id: str) -> BacterialTaxa:
        pass


class OrganismDataset(Dataset):
    def __init__(
            self,
            db: BacterialTaxaDatabase,
            dataset_df: pd.DataFrame
    ):
        """
        Initialize the microbiome dataset.

        :param dataset_df: A dataframe containing the profile(s) to be included. Index should be
        sample IDs, and each column should be a taxonomic ID.
        :param marker_embedding:
        """
        super().__init__()
        profile_extractor = MetaphlanProfileCollection(dataset_df)
        self.profiles = list(profile_extractor.samples())
        self.profile_organisms: List[Sample] = [
            [db.fetch_taxa(sgb_id) for sgb_id in profile.sgb_ids]
            for profile in self.profiles
        ]

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        features = self.profile_organisms[idx]
        labels = torch.from_numpy(self.profiles[idx].abundances_ensure_normalized)
        return features, labels
