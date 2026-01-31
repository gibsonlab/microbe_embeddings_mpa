"""
This module contains the dataset implementations for loading pre-computed embeddings.

Note that this is a separate computation strategay than the parent module, which is meant for loading raw sequences
as input, and computing the embeddings on-the-fly during training.
"""
from .abundance_profile import MetaphlanProfile, MetaphlanProfileCollection
from .embeddings import MetaphlanMarkerEmbedding
from .dataset import AbstractMetaphlanDataset, MetaphlanDataset
from .dataset_memmap import MetaphlanDatasetMemmapped, MetaphlanDatasetMemmappedTensorDict, perform_allocation
from .dataset_hdf5 import MetaphlanHDF5Dataset, HDF5BatchShuffledSampler
from .dataset_memmap_large import MetaphlanDatasetMemmappedLarge
