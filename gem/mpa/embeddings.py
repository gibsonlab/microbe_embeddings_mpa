from typing import *
from pathlib import Path

import h5py
import pandas as pd
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
        self.marker_index = self.calculate_marker_index()
        self.max_num_markers = int(self.marker_index.groupby("SGB")['Marker'].count().max())
        self.print_diagnostic()

    def print_diagnostic(self):
        print("Database statistics:")
        print("# SGB = {}".format(len(pd.unique(self.marker_index.groupby("SGB")))))
        print("# Markers = {}".format(self.marker_index.shape[0]))
        print("Max. # markers = {}".format(self.max_num_markers))

    def calculate_marker_index(self) -> pd.DataFrame:
        """
        :return: A mapping of [SGB ID] -> [List of Marker IDs]
        """
        df_parts = []
        for part_dir in sorted(self.marker_embedding_basedir.glob("part*")):
            assert (part_dir / ".embed.DONE").exists(), f"Embedding for part ({part_dir.name}) was not finished (dir={part_dir})."
            df = pd.read_csv(part_dir / "index.tsv", sep='\t')

            # Parse the Marker ID name, encoded in a previous stage (1_preprocess/1_degap_alignments.py)
            first_split = df['Marker'].str.split(":").str
            df['Protein'] = first_split[0]
            second_split = first_split[1].str.split("__").str
            df['SGB'] = "SGB{}".format(second_split[0])
            df['Part'] = part_dir.name
            df_parts.append(df)
        return pd.concat(df_parts, ignore_index=True)

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
        section = self.marker_index.loc[self.marker_index['SGB'] == sgb_id]
        for (part_subdir, shard_idx), shard_section in section.groupby(["Part", "Shard"]):
            # Open the appropriate shard file.
            shard_path = self.marker_embedding_basedir / part_subdir / f"shard-{shard_idx}.h5"
            with h5py.File(shard_path, "r") as shard:
                for marker_id in shard_section['Marker']:
                    marker_embedding = torch.tensor(shard[marker_id][:], dtype=self.dtype)
                    yield marker_id, marker_embedding

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
