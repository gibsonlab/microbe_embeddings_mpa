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

        # Fill in the matrix
        N = len(id_ordering)
        matrix = np.zeros(shape=(N, N), dtype=int)
        print(f"Populating {N} x {N} distance matrix...")
        for (i, asv_i), (j, asv_j) in tqdm(itertools.combinations(enumerate(id_ordering), r=2), total=N*(N-1) // 2):
            _d = hamming_distance(aligned_sequences[asv_i], aligned_sequences[asv_j])
            matrix[i, j] = _d
            matrix[j, i] = _d
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