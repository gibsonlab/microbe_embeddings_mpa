from typing import List, Tuple
from pathlib import Path

from Bio import SeqIO
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, save_npz as sparse_save_npz, load_npz as sparse_load_npz


class ASVDistanceMatrixSparse:
    def __init__(self, id_ordering: List[str], matrix: csr_matrix):
        self.id_ordering = id_ordering
        self.asv_to_idx = {asv_id: i for i, asv_id in enumerate(self.id_ordering)}
        self.matrix = matrix

    @staticmethod
    def from_alignment_sparse(aln_fasta: Path, n_threads: int = 1) -> 'ASVDistanceMatrixSparse':
        """ SPARSE (csr) implementation of the distance matrix. """
        # for each index i, ensure that the nearest (smallest distance) j is stored.
        # Note: this may mean that some rows may have more than 1 entry
        # (e.g. a bunch of indices i having j as its nearest neighbor, but j's neighbor isn't any of these i's.)

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
        print(f"Populating SPARSE {N} x {N} distance matrix... (nearest-neighbor only)")

        def compute_nearest_neighbor(i: int, seq_array: np.ndarray) -> Tuple[int, int, int]:
            """Compute nearest neighbor for sequence i"""
            distances = (seq_array != seq_array[i:i + 1]).sum(axis=1)
            distances[i] = -1  # Exclude self
            best_j = np.argmax(distances)
            best_dist = distances[best_j]
            return i, best_j, best_dist

        # Parallel computation, note: n_jobs=-1 uses all cores
        results = []
        with tqdm(total=N) as pbar:
            for result in Parallel(n_jobs=n_threads, return_as='generator')(
                    delayed(compute_nearest_neighbor)(i, seq_array)
                    for i in range(N)
            ):
                results.append(result)
                pbar.update(1)  # Update as each job completes

        data = []  # must be deduplicated, since duplicates are summed!
        row = []  # may have duplicates
        col = []  # may have duplicates
        for i, j, dist in results:
            row.append(i)
            col.append(j)
            data.append(dist)

            row.append(j)
            col.append(i)
            data.append(dist)

        # deduplicate before CSR creation
        print("Deduplicating sparse entries...")
        import pandas as pd
        df = pd.DataFrame({'row': row, 'col': col, 'data': data})
        df_dedup = df.drop_duplicates(subset=['row', 'col'], keep='last')
        matrix = csr_matrix((df_dedup['data'], (df_dedup['row'], df_dedup['col'])), shape=(N, N))

        print("Done.")
        return ASVDistanceMatrixSparse(id_ordering, matrix)

    def save(self, path: Path):
        # np.savez(path, ids=np.array(self.id_ordering, dtype=object), matrix=self.matrix)
        sparse_save_npz(str(path), self.matrix)
        with open(path.parent / f'{path.stem}.ordering.txt', "wt") as out_f:
            for _id in self.id_ordering:
                out_f.write(f"{_id}\n")

    @staticmethod
    def load(path: Path) -> 'ASVDistanceMatrix':
        with open(path.parent / f'{path.stem}.ordering.txt', "rt") as out_f:
            id_ordering = out_f.read().splitlines()
            id_ordering = [_id for _id in id_ordering if len(_id) > 0]
        matrix = sparse_load_npz(path)
        return ASVDistanceMatrix(id_ordering, matrix)


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