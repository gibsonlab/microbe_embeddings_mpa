"""
This module contains the dataset implementations for loading pre-computed embeddings.

Note that this is a separate computation strategay than the "generic" submodule, which is meant for loading raw sequences
as input, and computing the embeddings on-the-fly during training.
"""
from .embeddings import MetaphlanMarkerPrecomputedEmbedding
from .dataset import AbstractMetaphlanPreembeddedDataset, MetaphlanPreembeddedDataset
from .dataset_memmap import MetaphlanPreembeddedDatasetMemmapped, MetaphlanPreembeddedDatasetMemmappedTensorDict, perform_allocation
from .dataset_hdf5 import MetaphlanHDF5PreembeddedDataset, HDF5BatchShuffledSampler
from .dataset_memmap_large import MetaphlanPreembeddedDatasetMemmappedLarge
