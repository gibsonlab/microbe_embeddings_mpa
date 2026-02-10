from abc import abstractmethod, ABC
from typing import List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
import atexit

from Bio import SeqIO
import numpy as np
from tqdm import tqdm


class ASVDistanceMatrix(ABC):
    @abstractmethod
    def seq_len(self) -> int:
        pass

    @abstractmethod
    def get_asv_ids(self) -> List[str]:
        pass

    @abstractmethod
    def contains_asv(self, asv_id: str) -> bool:
        pass

    @abstractmethod
    def get_asv_index(self, asv_id: str) -> int:
        pass

    @abstractmethod
    def submatrix(self, asv_indices_1: List[str], asv_indices_2: List[str]) -> np.ndarray:
        pass


class ASVDistanceMatrixLazy(ASVDistanceMatrix):
    def __init__(self, alignment_file: Path, n_workers: int, enable_progress_bar: bool = False):
        self.id_ordering, self.seqlen, self.alignments = self.parse_alignments(alignment_file)
        self.asv_to_idx = {asv_id: i for i, asv_id in enumerate(self.id_ordering)}
        self.n_workers = n_workers
        self.executor = None
        self.enable_progress_bar = enable_progress_bar

    def get_asv_ids(self) -> List[str]:
        return self.id_ordering

    @staticmethod
    def parse_alignments(aln_fasta: Path) -> Tuple[List[str], int, np.ndarray]:
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

        id_ordering = sorted(list(aligned_sequences.keys()))
        N = len(id_ordering)

        # Convert sequences to numeric array for vectorized operations
        print(f"Converting sequences to numeric array...")
        seq_length = len(aligned_sequences[id_ordering[0]])
        seq_array = np.zeros((N, seq_length), dtype=np.uint8)

        for i, asv_id in enumerate(id_ordering):
            seq_array[i] = np.frombuffer(aligned_sequences[asv_id].encode('ascii'), dtype=np.uint8)
        return id_ordering, seq_length, seq_array

    def contains_asv(self, asv_id: str) -> bool:
        return asv_id in self.asv_to_idx

    def get_asv_index(self, asv_id: str) -> int:
        return self.asv_to_idx[asv_id]

    def entry_async(self, asv_i: str, asv_j: str) -> Future:
        assert self.executor is not None, "ASVDistanceMatrixLazy must be initialized as a context manager (e.g. `with ASVDistanceMatrixLazy(...) as mat`)"
        future = self.executor.submit(self.alignment_hamming_dist, self.get_asv_index(asv_i), self.get_asv_index(asv_j))
        return future

    def seq_len(self) -> int:
        return self.seqlen

    def alignment_hamming_dist(self, asv_i_idx: int, asv_j_idx: int) -> int:
        aln_i = self.alignments[asv_i_idx]
        aln_j = self.alignments[asv_j_idx]
        return np.sum(aln_i != aln_j)

    def submatrix(self, asv_indices_1: List[str], asv_indices_2: List[str]) -> np.ndarray:
        n_rows = len(asv_indices_1)
        n_cols = len(asv_indices_2)

        # Create all the Future instances without blocking.
        jobs = [
            [
                self.entry_async(asv_indices_1[i], asv_indices_2[j])
                for j in range(n_cols)
            ]
            for i in range(n_rows)
        ]

        # Flatten jobs to track total progress
        total_jobs = n_rows * n_cols

        # Now, wait for the results.
        result = []
        if self.enable_progress_bar:
            with tqdm(total=total_jobs, desc="Computing matrix") as pbar:
                for i in range(n_rows):
                    row = []
                    for j in range(n_cols):
                        row.append(jobs[i][j].result())
                        pbar.update(1)
                    result.append(row)
        else:
            for i in range(n_rows):
                row = []
                for j in range(n_cols):
                    row.append(jobs[i][j].result())
                result.append(row)

        res = np.array(result, dtype=int)
        assert res.shape[0] == n_rows
        assert res.shape[1] == n_cols
        return res


    def shutdown(self):
        self.executor.shutdown(wait=True)

    # this class is implemented as a context.
    def __enter__(self):
        # multi-threaded impl
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
        atexit.register(self.executor.shutdown, wait=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Resource cleanup and optional exception handling goes here
        if exception_type:
            # Handle the exception if necessary
            print(f"An exception occurred: {exception_value}")
            # Returning True here would suppress the exception
            # Returning False (or nothing) will re-raise the exception

        # This is where your cleanup code should go
        print("Shutting down threads..")
        self.shutdown()


class ASVDistanceMatrixPrecomputed(ASVDistanceMatrix):
    def __init__(self, id_ordering: List[str], seqlen: int, matrix: np.ndarray):
        self.id_ordering = id_ordering
        self.asv_to_idx = {asv_id: i for i, asv_id in enumerate(self.id_ordering)}
        self.matrix = matrix
        self.seqlen = seqlen

    def get_asv_ids(self) -> List[str]:
        return self.id_ordering

    def contains_asv(self, asv_id: str) -> bool:
        return asv_id in self.asv_to_idx

    def get_asv_index(self, asv_id: str) -> int:
        return self.asv_to_idx[asv_id]

    def entry(self, asv_i: str, asv_j: str) -> int:
        return self.matrix[self.asv_to_idx[asv_i], self.asv_to_idx[asv_j]]

    def seq_len(self) -> int:
        return self.seqlen

    @staticmethod
    def from_alignment(aln_fasta: Path) -> 'ASVDistanceMatrix':
        # Parse multiple alignment output.
        print(f"Loading alignments from {aln_fasta}")
        aligned_sequences = {}
        with open(aln_fasta, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                aligned_sequences[record.id] = str(record.seq).upper()

        id_ordering = sorted(list(aligned_sequences.keys()))
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
        return ASVDistanceMatrix(id_ordering, seq_length, matrix)

    def submatrix(self, asv_indices_1: List[str], asv_indices_2: List[str]) -> np.ndarray:
        sample1_indices = [self.get_asv_index(asv_id) for asv_id in asv_indices_1]
        sample2_indices = [self.get_asv_index(asv_id) for asv_id in asv_indices_2]
        return self.matrix[np.ix_(sample1_indices, sample2_indices)]

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