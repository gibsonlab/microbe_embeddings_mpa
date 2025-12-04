from typing import *
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from tensordict import TensorDict


class MetaphlanMarkerEmbedding:
    def __init__(
            self,
            marker_embedding_memmap_dir: Path,
    ):
        # Load cached tensors. (memory-mapped tensordict)
        assert marker_embedding_memmap_dir.exists(), f"Specified marker embeddings {marker_embedding_memmap_dir} does not exist!"
        self.marker_embedding_memmap_dir = marker_embedding_memmap_dir
        self.embedding_tensors = TensorDict.load_memmap(str(self.marker_embedding_memmap_dir))

        # Print diagnostic.
        example_embedding = next(iter(self.embedding_tensors.values()))
        print("Loaded tensor embeddings from {} (genome embedding shape = {})".format(
            self.marker_embedding_memmap_dir,
            example_embedding.shape
        ))

        # Compute padding size.
        self.dtype = example_embedding.dtype
        self.padding_marker_embedding = torch.zeros_like(example_embedding)
        assert len(self.padding_marker_embedding.shape) == 1, f"Embedding should be a vector! Got shape {self.padding_marker_embedding.shape} instead."
        self.embedding_dim = self.padding_marker_embedding.shape[0]

        # Compute database mapping.
        self.sgb_to_markers = dict()
        embedding_index = pd.read_parquet(marker_embedding_memmap_dir / "embedding_index.parquet")
        for _, row in embedding_index.iterrows():
            sgb_id = "SGB{}".format(row['SGB'])
            marker_id = row['Marker']
            if sgb_id not in self.sgb_to_markers:
                self.sgb_to_markers[sgb_id] = list()
            self.sgb_to_markers[sgb_id].append(marker_id)
        self.max_num_markers = max(len(x) for x in self.sgb_to_markers.values())

    def convert_sgb(self, sgb_id: str, max_markers: int) -> Tuple[Tensor, Tensor]:
        """
        Convert SGB to tensor format. Thread-safe.
        """
        if sgb_id not in self.sgb_to_markers:
            raise KeyError(f"SGB ID `{sgb_id}` not found in embeddings memory-mapped cache.")

        # This guarantees that our tensors are large enough to hold everything.
        num_markers = len(self.sgb_to_markers[sgb_id])
        if num_markers > max_markers:
            raise ValueError(
                f"max_markers was {max_markers}, but SGB has {num_markers} markers!")

        # Initialize tensors with padding
        marker_embeddings = []

        # Fill in the tensors
        for marker_id in self.sgb_to_markers[sgb_id]:
            marker_embed = self.embedding_tensors[marker_id]
            marker_embeddings.append(marker_embed)

        # Create padding mask tensor. Also fill the embeddings with padding.
        marker_padding_mask = torch.zeros(max_markers, dtype=torch.bool)
        marker_padding_mask[:len(marker_embeddings)] = True  # True indicates that this position is NOT empty

        marker_embeddings += [self.padding_marker_embedding for _ in range(max_markers - len(marker_embeddings))]
        marker_embeddings = torch.stack(marker_embeddings, dim=0)
        return marker_embeddings, marker_padding_mask
