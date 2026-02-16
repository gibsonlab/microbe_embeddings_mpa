import json
import sys, logging
import argparse
import threading

import zstandard as zstd
from typing import *
from pathlib import Path
import itertools

from tqdm import tqdm
import numpy as np
import torch
from pyfaidx import Fasta

from gem.glms import GenomeEmbedding

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def parse_cuda_device_ids(cuda_device_ids: str) -> List[torch.device]:
    cuda_device_ids = [int(x) for x in cuda_device_ids.split(",") if len(x) > 0]
    if len(cuda_device_ids) == 0:
        print(f"At least one CUDA device ID must be specified. Got: {cuda_device_ids}")
        exit(1)

    cuda_devices = []
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Total CUDA devices available: {device_count}")

        for device_id in cuda_device_ids:
            if device_id < device_count:
                print(f"CUDA device :{device_id} exists and is available.")
                # You can now create a device object for it
                device = torch.device(f"cuda:{device_id}")
                cuda_devices.append(device)
            else:
                print(f"CUDA device :{device_id} does not exist. Only devices 0 to {device_count - 1} are available.")
                exit(1)
    else:
        print("CUDA is not available on this system.")

    assert len(cuda_devices) > 0, "Unexpected error: parsed zero CUDA devices."
    return cuda_devices


def get_model_fn(model_name: str) -> Tuple[Callable[[torch.device], GenomeEmbedding], int]:
    """
    Parse the model creation function.
    :return: A callable which takes a torch.device (specifying which CUDA/CPU device to run the embedding model on,
     and outputs the requested GenomeEmbedding instance. Also returns the (Expected/hard-coded) embed dimension.
    """
    if model_name.startswith("evo-1"):
        tokens = model_name.split(":")
        if len(tokens) == 1:
            evo_checkpoint_name = tokens[0]
            num_hyena_layers = 32
        elif len(tokens) == 2:
            evo_checkpoint_name = tokens[0]
            num_hyena_layers = int(tokens[-1])
        else:
            raise RuntimeError("Incorrect model name syntax. Expected '<evo_checkpoint_name>:<n_layers>', but got {} instead.".format(model_name))

        from gem.glms.evo import EvoWrapper
        model_fn = lambda device: EvoWrapper(device=device, num_hyena_layers=num_hyena_layers, checkpoint_name=evo_checkpoint_name)
        return model_fn, 4096
    elif model_name.startswith("evo2"):
        tokens = model_name.split(":")
        if len(tokens) == 1:
            evo2_checkpoint_name = tokens[0]
            num_hyena_layers = 32
        elif len(tokens) == 2:
            evo2_checkpoint_name = tokens[0]
            num_hyena_layers = int(tokens[-1])
        else:
            raise RuntimeError("Incorrect model name syntax. Expected '<evo2_checkpoint_name>:<n_layers>', but got {} instead.".format(model_name))

        from gem.glms.evo2 import Evo2Wrapper
        model_fn = lambda device: Evo2Wrapper(device=device, num_hyena_layers=num_hyena_layers, checkpoint_name=evo2_checkpoint_name)
        return model_fn, 4096
    elif model_name == 'dnabert-s':
        from gem.glms.dnabert import DNABertSWrapper
        model_fn = lambda device: DNABertSWrapper(device=device)
        return model_fn, 768
    else:
        raise ValueError("Unknown model name {}".format(model_name))


# ================================================
class MarkerIndex:
    """
    Class which loads a marker index catalogue. Gene/Marker sequences are stored in the FASTA file, and the mapping
    from SGB ID -> (List of seq IDs) is stored as a ZSTD-compressed json object.
    """
    def __init__(self, fasta_file: Path, json_zstd_catalogue: Path):
        with zstd.open(json_zstd_catalogue, "rt") as f:
            sgb_marker_index = json.load(f)

        self.sgb_marker_index = sgb_marker_index
        self.fasta_file = fasta_file
        self.fasta: Union[Fasta, None] = None

    def __contains__(self, sgb_id: str) -> bool:
        return sgb_id in self.sgb_marker_index

    def get_fasta_seq(self, seq_id: str) -> str:
        assert self.fasta is not None, "MarkerIndex must be accessed through a context."
        if seq_id not in self.fasta:
            raise KeyError("Sequence {} not in fasta resource {}.".format(
                seq_id, self.fasta_file
            ))
        return str(self.fasta[seq_id])

    def get_sgb_markers(self, sgb_id: str) -> List[str]:
        if sgb_id in self:
            return [
                self.get_fasta_seq(marker_id)
                for marker_id in self.sgb_marker_index[sgb_id]
            ]
        else:
            return []

    def num_sgb_markers(self, sgb_id: str) -> int:
        if sgb_id in self:
            return len(self.sgb_marker_index[sgb_id])
        else:
            return 0

    def __enter__(self):
        self.fasta = Fasta(self.fasta_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fasta.close()
        self.fasta = None


class CompoundMarkerIndex:
    def __init__(self, indices: List[MarkerIndex]):
        self.indices = indices

    def __contains__(self, sgb_id: str) -> bool:
        return any(sgb_id in index for index in self.indices)

    def get_sgb_markers(self, sgb_id: str) -> List[str]:
        return list(
            itertools.chain.from_iterable(
                index.get_sgb_markers(sgb_id)
                for index in self.indices
            )
        )

    def num_sgb_markers(self, sgb_id: str) -> int:
        return sum(
            index.num_sgb_markers(sgb_id)
            for index in self.indices
        )

    def __enter__(self):
        for index in self.indices:
            index.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for index in self.indices:
            index.__exit__(exc_type, exc_val, exc_tb)


def precompute_embeddings(
        model_fn: Callable[[torch.device], GenomeEmbedding],
        memmap_array: np.memmap,
        cuda_devices: List[torch.device],
        sgb_ids: List[str],
        sgb_marker_index: Union[MarkerIndex, CompoundMarkerIndex],
):
    n_workers = len(cuda_devices)
    print("Using cuda devices: {}".format(",".join(str(dev) for dev in cuda_devices)))
    print("n_workers = {}".format(n_workers))

    # Split work among workers
    sgb_ids_per_worker = len(sgb_ids) // n_workers
    worker_assignments = []
    worker_start_indices = []

    for worker_idx in range(n_workers):
        start_idx = worker_idx * sgb_ids_per_worker
        if worker_idx == n_workers - 1:
            # Last worker gets any remaining items
            end_idx = len(sgb_ids)
        else:
            end_idx = (worker_idx + 1) * sgb_ids_per_worker
        worker_sgb_ids = sgb_ids[start_idx:end_idx]
        worker_start_indices.append(start_idx)
        worker_assignments.append(worker_sgb_ids)

    # Shared progress bar and lock for HDF5 file access
    pbar = tqdm(total=len(sgb_ids))
    memmap_lock = threading.Lock()

    def worker_fn(_worker_idx: int, _idx_offset: int, _device: torch.device, _worker_sgb_ids: List[str]):
        """Worker function that processes a subset of ASV IDs on a specific device."""
        # Create model instance for this worker
        print(f"Initializing worker {_worker_idx} [SGB index {_idx_offset} -- {_idx_offset + len(_worker_sgb_ids) - 1} (inclusive)]...")
        embedding_model = model_fn(_device)
        print(f"Created embedding model {embedding_model.__class__.__name__} on worker {_worker_idx}")

        # Process one SGB at a time
        for sgb_idx, sgb_id in enumerate(_worker_sgb_ids):
            marker_seqs: List[str] = sgb_marker_index.get_sgb_markers(sgb_id)
            n_markers = len(marker_seqs)
            global_idx = _idx_offset + sgb_idx
            if n_markers > 0:
                marker_embeddings = embedding_model.embed_batch(marker_seqs)
                with memmap_lock:
                    memmap_array[global_idx, :n_markers, :] = marker_embeddings.cpu().numpy()
                    memmap_array[global_idx, n_markers:, :] = np.nan  # fill rest with padding value (np.nan)
                    memmap_array.flush()
                    pbar.update(1)
            else:
                with memmap_lock:
                    memmap_array[global_idx, :, :] = np.nan
                    memmap_array.flush()
                    pbar.update(1)

    """ Multithreading task start, distributed across GPUs. """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import sys
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = []
        for worker_idx in range(n_workers):
            device = cuda_devices[worker_idx]
            worker_sgb_ids = worker_assignments[worker_idx]
            offset = worker_start_indices[worker_idx]

            future = executor.submit(worker_fn, worker_idx, offset, device, worker_sgb_ids)
            futures.append(future)

        # Wait for completion and handle exceptions
        for future in as_completed(futures):
            try:
                future.result()  # This will raise if the worker raised
            except Exception as e:
                print(f"Worker crashed: {e}", file=sys.stderr)
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
    print("All workers finished.")
    pbar.close()



def do_job(
        model_name: str,
        cuda_devices: List[torch.device],
        sgb_subset: Set[str],
        sgb_marker_index: Union[MarkerIndex, CompoundMarkerIndex],
        output_path: Path,
):
    """
    :param model_name:
    :param cuda_devices: A list of torch CUDA devices.
    :param sgb_subset: A list of SGB ids to include. Only SGBs in this collection will be embedded.
    :param sgb_marker_index: A MarkerIndex object.
    :param output_path: Path to the output file, which is a numpy memmap array file.
    :return:
    """
    model_fn, expected_embed_dim = get_model_fn(model_name)
    max_num_markers = max(sgb_marker_index.num_sgb_markers(sgb_id) for sgb_id in sgb_subset)

    # Initialize empty memmap
    memmap_array = np.memmap(
        output_path,
        dtype='float32',
        mode='w+',
        shape=(len(sgb_subset), max_num_markers, expected_embed_dim)
    )

    precompute_embeddings(
        model_fn,
        memmap_array,
        cuda_devices,
        sorted(sgb_subset),
        sgb_marker_index,
    )
    del memmap_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, required=True)
    parser.add_argument('-s', '--sgb-subset-file', type=str, required=True)
    parser.add_argument('-idx', '--sgb-marker-index', dest='sgb_marker_indices', type=str, required=True, nargs='+')
    parser.add_argument('-c', '--cuda-device-ids', type=str, required=True)
    parser.add_argument(
        '-o', '--output-path', type=str, required=True,
        help='The output embedding file path. The format is a numpy memmap array file.'
    )
    # Add more args as needed for marker indices
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.sgb_subset_file, "rt") as f:
        _sgb_subset = {l.strip() for l in f if len(l.strip()) > 0}

    _cuda_devices = parse_cuda_device_ids(args.cuda_device_ids)

    index_dirs = []
    for dir_str in args.sgb_marker_indices:
        index_dir = Path(dir_str)
        assert index_dir.exists() and index_dir.is_dir(), f"Directory {dir_str} does not exist!"

        fasta_p = index_dir / 'markers.fna'
        json_p = index_dir / 'sgb_marker_index.json.zst'
        assert fasta_p.exists(), f"Fasta file {fasta_p.name} not found in index dir {dir_str}"
        assert json_p.exists(), f"JSON index file {json_p.name} not found in index dir {dir_str}"
        index_dirs.append((fasta_p, json_p))

    assert len(index_dirs) > 0, "Need at least one marker index."

    marker_indices: List[MarkerIndex] = [
        MarkerIndex(fasta_file=fasta_path, json_zstd_catalogue=json_path)
        for (fasta_path, json_path) in index_dirs
    ]

    with CompoundMarkerIndex(marker_indices) as compound_index:
        do_job(
            model_name=args.model_name,
            cuda_devices=_cuda_devices,
            sgb_subset=_sgb_subset,
            sgb_marker_index=compound_index,
            output_path=Path(args.output_path),
        )