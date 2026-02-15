from typing import *
from pathlib import Path

import numpy as np
from umap import UMAP
from Bio.Seq import Seq
from Bio import SeqIO
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
            np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
        )
    return ids, np.stack(seqs, axis=0)


def parse_fasta_str(fasta_path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    seqs: List[str] = []
    for seq_id, seq in parse_fasta_raw(fasta_path):
        ids.append(seq_id)
        seqs.append(str(seq))
    return ids, seqs


class UMAPEmbedding(GenomeEmbedding):
    """
    Class which embeds a batch of pre-aligned sequences using UMAP.

    Unlike Evo/DNABERT, this is NOT a "genome language model", rather it's an embedding using a pre-computed multiple
    alignment. The only embeddings allowed are those sequences passed into the original instantiation.
    (note: it's also possible to use a 'parametric UMAP' and fine-tune it for new examples, but that depends heavily on
    a train-test split and is less likely to produce good embeddings.)

    Embedding of "sequences" is done by hashing: the raw nucleotide sequence is provided as a lookup key to the
    original umap fitting training set.

    The distance metric is the hamming distance between aligned sequences.
    Thus, this embedding only makes sense for homologous sequences (e.g. ASVs from the same amplicon region).
    """
    def __init__(self, unaligned_fasta: Path, multi_alignment_fasta: Path, embed_dim: int, rng_seed: int, device: torch.device):
        """
        :param unaligned_fasta: The sequences that will be used to query this model.
        :param multi_alignment_fasta: The multiple alignments used to compute hamming distances for UMAP's kNN queries.
        :param embed_dim: The target output embedding dim.
        :param rng_seed: The seed to use for UMAP fitting (UMAP is a randomized embedding).
        :param device: The output torch device to instantiate the embeddings on.
        """
        self.device = device
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

        umap_model = UMAP(random_state=rng_seed, n_components=embed_dim, metric='hamming')
        umap_model.fit(aln_seq_array)
        embeddings: np.ndarray = umap_model.transform(aln_seq_array)  # numpy array, shape (N, embed_dim)

        self.seq_embeddings: Dict[str, np.ndarray] = {
            raw_seq: embeddings[aln_indices[raw_seq_id]]
            for (raw_seq_id, raw_seq) in zip(raw_seq_ids, raw_sequences)
        }

    def device(self) -> torch.device:
        return self.device

    def embed_dim(self) -> int:
        return self.embed_dim

    def embed_sequence(self, x: str) -> Tensor:
        try:
            embedding = self.seq_embeddings[x]
            return torch.from_numpy(embedding).to(self.device)
        except KeyError:
            raise KeyError(f"Sequence {x} not found in non-parametric UMAP embedding training set.") from None

    def embed_batch(self, strs: List[str]) -> Tensor:
        return torch.stack([
            self.embed_sequence(x) for x in strs
        ], dim=0)
