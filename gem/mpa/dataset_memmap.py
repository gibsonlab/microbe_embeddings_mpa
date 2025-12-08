from typing import *

from pathlib import Path
import pandas as pd
from torch import Tensor
from tensordict import TensorDict
from tqdm import tqdm

from .embeddings import MetaphlanMarkerEmbedding
from .dataset import MetaphlanDataset

"""
From https://docs.pytorch.org/tensordict/main/saving.html

1) Saving a memmapped tensordict:
    x = TensorDict()
    x_disk = x.memmap("/path/to/saved/dir", num_threads=30)

2) Loading a memmapped tensordict:
    x = TensorDict.load_memmap("/path/to/saved/dir")
"""

class MetaphlanDatasetMemmapped(MetaphlanDataset):
    """
    A class which pre-computes all tensors and stores into a memory-mapped tensordict.
    """

    def __init__(
            self,
            dataset_df: pd.DataFrame,
            marker_embedding: MetaphlanMarkerEmbedding,
            cache_dir: Path
    ):
        super().__init__(dataset_df, marker_embedding)
        self.cache_dir = cache_dir
        print(f"Using tensor memmap directory: {cache_dir}")

        self.tensor_cache: List[TensorDict] = []
        self.allocate_memmap_tensors()

    def allocate_memmap_tensors(self):
        for sample_idx, sample in enumerate(tqdm(self.samples, desc="Sample Allocation")):
            memmap_dir = self.cache_dir / sample.sample_id
            if (memmap_dir / "meta.json").exists():
                # TensorDict is already allocated; load it from disk.
                x = TensorDict.load_memmap(memmap_dir)
            else:
                # Allocate the TensorDict.
                memmap_dir.mkdir(exist_ok=True, parents=False)  # parent dir should already exist!
                features, marker_padding_mask, sgb_padding_mask, targets = super().__getitem__(sample_idx)
                x = TensorDict()
                x['features'] = features
                x['mpadding'] = marker_padding_mask
                x['spadding'] = sgb_padding_mask
                x['targets'] = targets
                x.memmap(str(memmap_dir))
            self.tensor_cache.append(x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Load from pre-computed tensordict.
        """
        x = self.tensor_cache[idx]
        return x['targets'], x['mpadding'], x['spadding'], x['targets']
