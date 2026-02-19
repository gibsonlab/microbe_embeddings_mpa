from typing import *
from pathlib import Path

import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
import torch
from torch import Tensor

from .base import GenomeEmbedding


def parse_fasta_raw(fasta_path: Path) -> Generator[Tuple[str, Seq], None, None]:
    with open(fasta_path, "rt") as f:
        for record in SeqIO.parse(f, "fasta"):
            yield record.id, record.seq


def parse_fasta_numpy(fasta_path: Path) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []
    seqs: List[np.ndarray] = []
    for seq_id, seq in parse_fasta_raw(fasta_path):
        ids.append(seq_id)
        seqs.append(
            np.frombuffer(str(seq).upper().encode('ascii'), dtype=np.uint8)
        )
    return ids, np.stack(seqs, axis=0)


def parse_fasta_str(fasta_path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    seqs: List[str] = []
    for seq_id, seq in parse_fasta_raw(fasta_path):
        ids.append(seq_id)
        seqs.append(str(seq))
    return ids, seqs


def hamming_distance_matrix(alignments: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Given an (N x d) feature vector array (e.g. each feature vector is the aligned nucleotide seqs)
    output an N x N pairwise hamming-distance matrix.
    :param alignments:
    :return:
    """
    print("Number of aligned seqs: {}, align_len = {}".format(alignments.shape[0], alignments.shape[1]))
    print(f"Computing hamming distance matrix with chunk_len = {chunk_size}")
    N = alignments.shape[0]
    hamming_distances = np.zeros((N, N), dtype=np.int32)  # use int32 instead of int64 if possible

    # Compute in chunks to avoid memory issues
    from tqdm import tqdm
    for i in tqdm(range(0, N, chunk_size)):
        i_end = min(i + chunk_size, N)
        chunk = alignments[i:i_end]

        # Compare this chunk against all alignments
        # Shape: (chunk_size, N, d) but computed efficiently
        hamming_distances[i:i_end, :] = (chunk[:, None, :] != alignments[None, :, :]).sum(axis=2)

    return hamming_distances


class PCoAEmbedding(GenomeEmbedding):
    """
    Class which embeds a batch of pre-aligned sequences using PCoA, evaluated using the hamming distance matrix
    derived from multiple sequence alignment of ASVs.

    Unlike Evo/DNABERT, this is NOT a "genome language model", rather it's an embedding using a pre-computed multiple
    alignment. The only embeddings allowed are those sequences passed into the original instantiation.

    Embedding of "sequences" is done by hashing: the raw nucleotide sequence is provided as a lookup key to the
    original ASV fitting training set.
    Since the distance metric is hamming distance of multiple alignments, this embedding only makes sense for
    homologous sequences (e.g. ASVs from the same amplicon region).
    """
    def __init__(self, unaligned_fasta: Path, multi_alignment_fasta: Path, embed_dim: int, rng_seed: int, chunk_size: int):
        """
        :param unaligned_fasta: The sequences that will be used to query this model.
        :param multi_alignment_fasta: The multiple alignments used to compute hamming distances for UMAP's kNN queries.
        :param embed_dim: The target output embedding dim.
        :param rng_seed: The seed to use for UMAP fitting (UMAP is a randomized embedding).
        :param chunk_size: the chunk size to ues for computing the pairwise hamming distance matrix. Smaller = slower, but smaller memory footprint.
        """
        self.embed_dim = embed_dim
        self.rng_seed = rng_seed

        # N = the number of sequences.
        aln_seq_ids, aln_seq_array = parse_fasta_numpy(multi_alignment_fasta)
        raw_seq_ids, raw_sequences = parse_fasta_str(unaligned_fasta)

        aln_indices = {s_id: _i for _i, s_id in enumerate(aln_seq_ids)}
        for raw_seq_id, raw_seq in zip(raw_seq_ids, raw_sequences):
            if raw_seq_id not in aln_indices:
                raise KeyError("Sequence FASTA record `{}` not found in alignment file `{}`.".format(
                    raw_seq_id, multi_alignment_fasta.name
                ))

        distance_matrix = DistanceMatrix(
            data=hamming_distance_matrix(aln_seq_array, chunk_size=chunk_size),
            ids=aln_seq_ids,
        )

        print(f"Running PCoA embedding. dim = {embed_dim}, seed = {rng_seed}")
        pcoa_results = pcoa(distance_matrix, dimensions=embed_dim, seed=rng_seed)
        coordinates = pcoa_results.samples.values

        self.seq_embeddings: Dict[str, np.ndarray] = {
            raw_seq: coordinates[aln_indices[raw_seq_id], :]
            for (raw_seq_id, raw_seq) in zip(raw_seq_ids, raw_sequences)
        }

    def device(self) -> torch.device:
        return torch.device("cpu")

    def embed_dim(self) -> int:
        return self.embed_dim

    def embed_sequence(self, x: str) -> Tensor:
        try:
            embedding = self.seq_embeddings[x]
            return torch.from_numpy(embedding)  # cpu tensor, float32
        except KeyError:
            raise KeyError(f"Sequence {x} not found in non-parametric PCoA embedding training set.") from None

    def embed_batch(self, strs: List[str]) -> Tensor:
        return torch.stack([
            self.embed_sequence(x) for x in strs
        ], dim=0)
