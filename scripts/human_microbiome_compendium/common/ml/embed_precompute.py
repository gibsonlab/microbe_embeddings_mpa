from typing import *
from pathlib import Path

import threading
from tqdm import tqdm
import h5py
import torch

from gem.glms import GenomeEmbedding


# def precompute_embeddings(
#         asv_seqs: Dict[str, str],
#         embed_create_fn: Callable[[torch.device], GenomeEmbedding],
#         hdf5_output_path: Path,
#         batch_size: int,
#         cuda_device_list: List[torch.device],
# ):
#     """
#     Embed the ASV sequences and save each embedding as a separate dataset in an HDF5 file.
#     :param asv_seqs:
#     :param embed_create_fn:
#     :param hdf5_output_path:
#     :param batch_size:
#     :param num_workers:
#     :return:
#     """
#     pbar = tqdm(total=len(asv_seqs))
#     asv_id_list = sorted(asv_seqs.keys())
#
#     n_workers = 0  # should create as many workers as devices in cuda_device_list
#     embedding_model = embed_create_fn(cuda_device_list[0])  # should create as many model instances as workers.
#
#     with h5py.File(hdf5_output_path, 'w') as f:
#         for i in range(0, len(asv_id_list), batch_size):  # compute embedding in batches.
#             asv_id_batch = asv_id_list[i:i + batch_size]
#             asv_seq_batch = [asv_seqs[asv_id] for asv_id in asv_id_batch]
#             batch_embeddings = embedding_model.embed_batch(asv_seq_batch).cpu().float().numpy()  # shape (batch_len, embed_dim)
#             for asv_id, asv_embedding in zip(asv_id_batch, batch_embeddings):  # each ASV_ID gets its own entry.
#                 f.create_dataset(asv_id, data=asv_embedding, compression='lzf')
#             pbar.update(len(asv_id_batch))


def precompute_embeddings(
        asv_seqs: Dict[str, str],
        embed_create_fn: Callable[[torch.device], GenomeEmbedding],
        hdf5_output_path: Path,
        batch_size: int,
        cuda_device_list: List[torch.device],
):
    """
    Embed the ASV sequences and save each embedding as a separate dataset in an HDF5 file.
    Uses multi-threading with one worker per CUDA device.

    :param asv_seqs: Dictionary mapping ASV IDs to sequences
    :param embed_create_fn: Function that creates a GenomeEmbedding model on a given device
    :param hdf5_output_path: Path to output HDF5 file
    :param batch_size: Batch size for embedding computation
    :param cuda_device_list: List of CUDA devices to use
    """
    asv_id_list = sorted(asv_seqs.keys())

    print("Using cuda devices: {}".format(
        ",".join(str(dev) for dev in cuda_device_list)
    ))

    n_workers = len(cuda_device_list)
    print("n_workers = {}".format(n_workers))

    # Create empty HDF5 file with pre-allocated structure
    with h5py.File(hdf5_output_path, 'w') as _:
        print("Created HD5 file: {}".format(hdf5_output_path))
        pass  # Just create the file

    # Split work among workers
    asv_ids_per_worker = len(asv_id_list) // n_workers
    worker_assignments = []

    for worker_idx in range(n_workers):
        start_idx = worker_idx * asv_ids_per_worker
        if worker_idx == n_workers - 1:
            # Last worker gets any remaining items
            end_idx = len(asv_id_list)
        else:
            end_idx = (worker_idx + 1) * asv_ids_per_worker

        worker_asv_ids = asv_id_list[start_idx:end_idx]
        worker_assignments.append(worker_asv_ids)

    # Shared progress bar and lock for HDF5 file access
    pbar = tqdm(total=len(asv_id_list))
    hdf5_lock = threading.Lock()

    def worker_fn(_worker_idx: int, _device: torch.device, _worker_asv_ids: List[str]):
        """Worker function that processes a subset of ASV IDs on a specific device."""
        # Create model instance for this worker
        print(f"Initializing worker {_worker_idx} ({len(_worker_asv_ids)} ASVs to embed)...")
        embedding_model = embed_create_fn(_device)
        print(f"Created embedding model {embedding_model.__class__.__name__} on worker {_worker_idx}")

        # Process in batches
        for i in range(0, len(_worker_asv_ids), batch_size):
            asv_id_batch = _worker_asv_ids[i:i + batch_size]
            asv_seq_batch = [asv_seqs[asv_id] for asv_id in asv_id_batch]

            # Compute embeddings on this device
            batch_embeddings = embedding_model.embed_batch(asv_seq_batch).to("cpu").float().numpy()

            # Write to HDF5 file (thread-safe with lock)
            with hdf5_lock:
                with h5py.File(hdf5_output_path, 'a') as f:
                    for asv_id, asv_embedding in zip(asv_id_batch, batch_embeddings):
                        f.create_dataset(asv_id, data=asv_embedding, compression='lzf')

            # Update progress bar (thread-safe)
            pbar.update(len(asv_id_batch))

    # Create and start worker threads
    threads = []
    for worker_idx in range(n_workers):
        device = cuda_device_list[worker_idx]
        worker_asv_ids = worker_assignments[worker_idx]

        thread = threading.Thread(
            target=worker_fn,
            args=(worker_idx, device, worker_asv_ids)
        )
        thread.start()
        threads.append(thread)

    # Wait for all workers to complete
    for thread in threads:
        thread.join()
    print("All workers finished.")

    pbar.close()
