from typing import *
from pathlib import Path
from itertools import islice

import h5py
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


def chunk_matrix_generator(gen, chunk_size=5000):
    """Yields chunks of chunk_size from generator as lists."""
    iterator = iter(gen)
    while True:
        chunk = list(m for _, m in islice(iterator, chunk_size))
        if not chunk:
            break
        yield np.stack(chunk, axis=0)  # Shape: (chunk_size, feature_dim)


class MetaphlanMarkerEmbedding:
    """
    A class which encapsulates pre-computed embeddings (step 2: 2_embed/compute_embeddings.py) which automatically
    stores all marker embeddings in a sharded manner.
    """
    def __init__(
            self,
            marker_embedding_basedir: Path,
            dimension_reduce_pca: Optional[int] = None,
            ipca_batch_size: Optional[int] = None,
    ):
        assert marker_embedding_basedir.exists(), f"Specified marker embeddings {marker_embedding_basedir} does not exist!"
        self.marker_embedding_basedir = marker_embedding_basedir

        # Compute database mapping.
        self.marker_index = self.calculate_marker_index()
        self.num_markers_by_sgb = {
            sgb_id: int(count)
            for sgb_id, count in self.marker_index.groupby("SGB")['Marker'].count().items()
        }
        self.max_num_markers = max(self.num_markers_by_sgb.values())
        self.print_diagnostic()

        # Determine embedding dimension, and print diagnostic.
        self.apply_dimension_reduction = (dimension_reduce_pca is not None)
        if self.apply_dimension_reduction:
            self.embedding_dim = dimension_reduce_pca
            assert ipca_batch_size is not None, "If applying dimensionality reduction on embeddings, ipca_batch_size cannot be NoneType."
            self.pca_model, self.standard_scaler = self.dimension_reduce_embeddings(
                n_components=dimension_reduce_pca,
                ipca_batch_size=ipca_batch_size
            )
            print("Tensor embeddings source: {} (genome embedding shape = {} --> {} after PCA)".format(
                self.marker_embedding_basedir,
                self.embedding_dim,
                dimension_reduce_pca,
            ))
        else:
            example_embedding = self.get_raw_example_tensor()
            self.embedding_dim = example_embedding.shape[0]
            self.pca_model, self.standard_scale = None, None
            print("Tensor embeddings source: {} (genome embedding shape = {}, no PCA)".format(
                self.marker_embedding_basedir,
                self.embedding_dim
            ))

        # Compute padding size.
        self.padding_marker_embedding = np.zeros(shape=(self.embedding_dim,))
        assert len(self.padding_marker_embedding.shape) == 1, f"Embedding should be a vector! Got shape {self.padding_marker_embedding.shape} instead."

    @property
    def total_num_markers(self) -> int:
        return self.marker_index.shape[0]

    def print_diagnostic(self):
        print("Database statistics:")
        print("# SGB = {}".format(len(pd.unique(self.marker_index["SGB"]))))
        print("# Markers = {}".format(self.marker_index.shape[0]))
        print("Max. # markers (per SGB) = {}".format(self.max_num_markers))

    def calculate_marker_index(self) -> pd.DataFrame:
        """
        :return: A mapping of [SGB ID] -> [List of Marker IDs]
        """
        df_parts = []
        print(f"Loading embedding metadata from {self.marker_embedding_basedir}")
        for part_dir in sorted(self.marker_embedding_basedir.glob("part*")):
            print(f"Reading: {part_dir}")
            assert (part_dir / ".embed.DONE").exists(), f"Embedding for part ({part_dir.name}) was not finished (dir={part_dir})."
            df = pd.read_csv(part_dir / "index.tsv", sep='\t')

            # Parse the Marker ID name, encoded in a previous stage (1_preprocess/1_degap_alignments.py)
            first_split = df['Marker'].str.split(":").str
            df['Protein'] = first_split[0]
            second_split = first_split[1].str.split("__").str
            df['SGB'] = "SGB" + second_split[0]
            df['Part'] = part_dir.name
            df_parts.append(df)
        return pd.concat(df_parts, ignore_index=True)

    def num_markers(self, sgb_id: str) -> int:
        return self.num_markers_by_sgb[sgb_id]

    def get_raw_example_tensor(self) -> np.ndarray:
        """ Return an embedding tensor, without any dimensionality reduction. """
        part_dir = self.marker_embedding_basedir / "part1"
        with h5py.File(part_dir / "shard-0.h5", "r") as shard:
            first_key = next(iter(shard.keys()))
            example = shard[first_key][:]
            return example

    def _all_markers_raw(self, shuffle_shard: bool = False) -> Iterator[Tuple[str, np.ndarray]]:
        for part_dir in sorted(self.marker_embedding_basedir.glob("part*")):
            for shard_file in part_dir.glob("shard-*.h5"):
                with h5py.File(shard_file, "r") as shard:
                    marker_ids = list(shard.keys())
                    if shuffle_shard:
                        random.shuffle(marker_ids)

                    for marker_id in marker_ids:
                        marker_embedding_numpy = shard[marker_id][:]
                        yield marker_id, marker_embedding_numpy

    def get_sgb_markers_from_file(self, sgb_id: str) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Load markers from pre-computed embedding file.
        """
        section = self.marker_index.loc[self.marker_index['SGB'] == sgb_id]
        for (part_subdir, shard_idx), shard_section in section.groupby(["Part", "Shard"]):
            # Open the appropriate shard file.
            shard_path = self.marker_embedding_basedir / part_subdir / f"shard-{shard_idx}.h5"
            with h5py.File(shard_path, "r") as shard:
                for marker_id in shard_section['Marker']:
                    marker_embedding = shard[marker_id][:]
                    if self.apply_dimension_reduction:
                        vect = marker_embedding - self.standard_scale.mean_
                        vect = vect / self.standard_scale.scale_
                        yield marker_id, self.pca_model.transform(vect.reshape(1, -1))[0]
                    else:
                        yield marker_id, marker_embedding

    def convert_sgb(self, sgb_id: str, max_markers: int) -> Tuple[np.ndarray, np.ndarray]:
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
        marker_padding_mask = np.zeros(max_markers, dtype=bool)
        marker_padding_mask[:len(marker_embeddings)] = True  # True indicates that this position is NOT empty

        marker_embeddings += [self.padding_marker_embedding for _ in range(max_markers - len(marker_embeddings))]
        marker_embeddings = np.stack(marker_embeddings, axis=0)
        return marker_embeddings, marker_padding_mask

    def dimension_reduce_embeddings(
            self,
            n_components: int,
            ipca_batch_size: int,
    ) -> Tuple[IncrementalPCA, StandardScaler]:
        # Initialize
        ipca = IncrementalPCA(n_components=n_components, batch_size=ipca_batch_size)
        scaler = StandardScaler(with_mean=True, with_std=True)  # Centers + scales
        standardization_chunk_size = ipca_batch_size * 5

        # Step 1: Compute GLOBAL stats across ALL data
        for chunk_matrix in chunk_matrix_generator(
                gen=tqdm(self._all_markers_raw(), total=self.total_num_markers, desc="[Embedding] Standardization"),
                chunk_size=standardization_chunk_size
        ):  # Yield (chunk_size, 4096) arrays
            scaler.partial_fit(chunk_matrix)  # Updates running mean/var
            break

        # Step 2: Standardize and compute PCA.
        for chunk_matrix in chunk_matrix_generator(
                gen=tqdm(self._all_markers_raw(), total=self.total_num_markers, desc="[Embedding] Incremental-PCA"),
                chunk_size=standardization_chunk_size
        ):  # Yield (chunk_size, 4096) arrays
            # Apply GLOBAL standardization
            chunk_centered = chunk_matrix - scaler.mean_  # Subtract global mean
            chunk_scaled = chunk_centered / scaler.scale_  # Divide by global std
            ipca.partial_fit(chunk_scaled)
            break

        print(f"Embedding-PCA -- Explained variance ratio: {ipca.explained_variance_ratio_.sum():.3f}")
        return ipca, scaler
