from typing import *
from pathlib import Path

import h5py
import torch
from torch import Tensor


class MetaphlanMarkerEmbedding:
    def __init__(
            self,
            marker_embedding_basedir: Path,
            embed_dtype=torch.float32
    ):
        # Load cached tensors. (memory-mapped tensordict)
        assert marker_embedding_basedir.exists(), f"Specified marker embeddings {marker_embedding_basedir} does not exist!"
        self.marker_embedding_basedir = marker_embedding_basedir
        self.sgb_to_markers = dict()  # Mapping of [SGB ID] -> [List of Marker IDs]

        # Print diagnostic.
        example_embedding = self.get_example_tensor()
        print("Loaded tensor embeddings from {} (genome embedding shape = {})".format(
            self.marker_embedding_basedir,
            example_embedding.shape
        ))

        # Compute padding size.
        self.dtype = embed_dtype
        self.padding_marker_embedding = torch.zeros(size=example_embedding.shape)
        assert len(self.padding_marker_embedding.shape) == 1, f"Embedding should be a vector! Got shape {self.padding_marker_embedding.shape} instead."
        self.embedding_dim = self.padding_marker_embedding.shape[0]

        # Compute database mapping.
        self.max_num_markers = max(len(x) for x in self.sgb_to_markers.values())

    def get_example_tensor(self) -> Tensor:
        part_dir = self.marker_embedding_basedir / "part1"
        with h5py.File(part_dir / "shard-0.h5", "r") as shard:
            first_key = next(iter(shard.keys()))
            example = shard[first_key][:]
            return example

    def get_sgb_markers_from_file(self, sgb_id: str) -> Iterator[Tuple[str, Tensor]]:
        """
        Load markers from pre-computed embedding file.
        """
        n_markers_found = 0
        for part_dir in sorted(self.marker_embedding_basedir.glob("part*")):
            assert (part_dir / "embed.DONE").exists(), f"Embedding for part ({part_dir.name}) was not finished."
            current_shard_idx = -1
            current_shard_file = None

            try:
                with open(part_dir / "index.tsv", "rt") as index_file:
                    for line in index_file:
                        if not line.startswith(f"{sgb_id}__"):
                            continue

                        # found SGB marker token.
                        n_markers_found += 1
                        line_tokens = line.strip().split("\t")
                        marker_id = line_tokens[0]
                        marker_shard_idx = int(line_tokens[-1])

                        # Ensure that we have the correct shard open.
                        if current_shard_idx != marker_shard_idx:
                            # close the previous file, open a new one.
                            if current_shard_file is not None:
                                current_shard_file.close()
                            current_shard_file = h5py.File(part_dir / f"shard-{marker_shard_idx}.h5", "r")
                            current_shard_idx = marker_shard_idx

                        marker_embedding = torch.tensor(current_shard_file[marker_id][:], dtype=self.dtype)
                        yield marker_id, marker_embedding
            finally:
                # Ensure file is closed even if exception occurs
                if current_shard_file is not None:
                    current_shard_file.close()

    def convert_sgb(self, sgb_id: str, max_markers: int) -> Tuple[Tensor, Tensor]:
        """
        Convert SGB to tensor format. Thread-safe.
        """
        ### ===== version 2
        # Initialize tensors with padding
        marker_embeddings = []

        # Fill in the tensors
        for m_idx, (_, sgb_marker_embedding) in enumerate(self.get_sgb_markers_from_file(sgb_id)):
            marker_embeddings.append(sgb_marker_embedding)

        # This guarantees that our tensors are large enough to hold everything.
        num_markers = len(marker_embeddings)
        if num_markers > max_markers:
            raise ValueError(f"max_markers was {max_markers}, but SGB has {num_markers} markers!")

        # Create padding mask tensor. Also fill the embeddings with padding.
        marker_padding_mask = torch.zeros(max_markers, dtype=torch.bool)
        marker_padding_mask[:len(marker_embeddings)] = True  # True indicates that this position is NOT empty

        marker_embeddings += [self.padding_marker_embedding for _ in range(max_markers - len(marker_embeddings))]
        marker_embeddings = torch.stack(marker_embeddings, dim=0)
        return marker_embeddings, marker_padding_mask
