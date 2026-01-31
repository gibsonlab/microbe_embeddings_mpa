"""
This module contains the dataset implementations for loading Nucleotide/AA sequences as input, and computes embeddings
on-the-fly.
"""
from .taxa_db import BacterialTaxaDatabase, MetaphlanTaxaDatabase
from .dataset import OrganismGeneSequenceDataset
