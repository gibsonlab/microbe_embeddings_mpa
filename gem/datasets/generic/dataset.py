from typing import *

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..abundance_profile import AbundanceProfileParser
from .taxa_db import BacterialTaxaDatabase
from .types import *


def parse_all_samples(db, profile_parser, dtype: torch.dtype) -> Iterator[Tuple[Sample, Tensor]]:
    for profile in profile_parser.samples():
        # filter out SGBs that don't have any markers.
        organisms_subset: List[BacterialTaxa] = [
            db.fetch_taxa(taxa_id)
            for taxa_id in profile.taxa_ids
            if taxa_id in db
        ]

        targets = torch.tensor([
            abundance
            for taxa_id, abundance in zip(profile.taxa_ids, profile.abundances)
            if taxa_id in db
        ], dtype=dtype)
        yield (profile.sample_id, organisms_subset), targets / targets.sum()


class OrganismGeneSequenceDataset(Dataset):
    def __init__(
            self,
            db: BacterialTaxaDatabase,
            profile_parser: AbundanceProfileParser,
            dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the microbiome dataset.

        :param db: BacterialTaxaDatabase which can translate the taxa IDs found in profile_parser into lists
        of sequences.
        :param profile_parser: A parser that returns the profile(s) to be included.
        """
        super().__init__()
        self.samples = list(parse_all_samples(db, profile_parser, dtype))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[Sample, Tensor]:
        return self.samples[idx]
