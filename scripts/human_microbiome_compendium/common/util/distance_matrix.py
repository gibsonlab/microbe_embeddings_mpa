from typing import List
from pathlib import Path
import itertools

from Bio import SeqIO
import numpy as np
from tqdm import tqdm


class ASVDistanceMatrix:
    def __init__(self, id_ordering: List[str], matrix: np.ndarray):
        self.id_ordering = id_ordering
        self.asv_to_idx = {asv_id: i for i, asv_id in enumerate(self.id_ordering)}
        self.matrix = matrix

    def contains_asv(self, asv_id: str) -> bool:
        return asv_id in self.asv_to_idx

    def get_asv_index(self, asv_id: str) -> int:
        return self.asv_to_idx[asv_id]

    def entry(self, asv_i: str, asv_j: str) -> int:
        return self.matrix[self.asv_to_idx[asv_i], self.asv_to_idx[asv_j]]

    @staticmethod
    def from_alignment(aln_fasta: Path) -> 'ASVDistanceMatrix':
        # Parse multiple alignment output.
        print(f"Loading alignments from {aln_fasta}")
        aligned_sequences = {}
        with open(aln_fasta, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                aligned_sequences[record.id] = str(record.seq).upper()

        id_ordering = sorted(aligned_sequences.keys())
        N = len(id_ordering)

        # Convert sequences to numeric array for vectorized operations
        print(f"Converting sequences to numeric array...")
        seq_length = len(aligned_sequences[id_ordering[0]])
        seq_array = np.zeros((N, seq_length), dtype=np.uint8)

        for i, asv_id in enumerate(id_ordering):
            seq_array[i] = np.frombuffer(aligned_sequences[asv_id].encode('ascii'), dtype=np.uint8)

        # Vectorized hamming distance calculation
        print(f"Populating {N} x {N} distance matrix...")
        matrix = np.zeros((N, N), dtype=int)

        for i in tqdm(range(N)):
            # Calculate distances from sequence i to all sequences at once
            diffs = seq_array != seq_array[i:i + 1]
            distances = diffs.sum(axis=1)
            matrix[i] = distances

        print("Done.")
        return ASVDistanceMatrix(id_ordering, matrix)

    def save(self, path: Path):
        np.savez(path, ids=np.array(self.id_ordering, dtype=object), matrix=self.matrix)

    @staticmethod
    def load(path: Path) -> 'ASVDistanceMatrix':
        print(f"Loading matrix from file {path}")
        data = np.load(path, allow_pickle=True)
        id_ordering = list(data['ids'])
        matrix = data['matrix']
        return ASVDistanceMatrix(id_ordering, matrix)


def hamming_distance(seq1: str, seq2: str) -> int:
    assert len(seq1) == len(seq2), "For Hamming distance, sequences must be of equal length after alignment."
    distance = sum(1 for x, y in zip(seq1, seq2) if x != y)
    return distance