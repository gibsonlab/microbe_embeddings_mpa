from typing import *
from pathlib import *
import argparse

import torch
from Bio import SeqIO

from gem.glms import GenomeEmbedding
from common.ml import precompute_embeddings


def load_fasta_dict(fasta_path: Path) -> Dict[str, str]:
    return {
        record.id: str(record.seq)
        for record in SeqIO.parse(fasta_path, "fasta")
    }


def main(
        asv_fasta_file: Path,
        hdf5_output_path: Path,
        model_name: str,
        embed_batch_size: int,
        cuda_devices: List[torch.device],
):
    """
    :param asv_fasta_file: Path to a multi-FASTA nucleotide sequence file, where each record is
    the consensus sequence of an ASV.
    :param hdf5_output_path:
    :param model_name:
    :param embed_batch_size:
    :param cuda_devices:
    :return:
    """
    asv_seqs = load_fasta_dict(asv_fasta_file)
    embed_create_fn = embedding_model_initializer(model_name)
    precompute_embeddings(asv_seqs, embed_create_fn, hdf5_output_path, embed_batch_size, cuda_devices)


def embedding_model_initializer(model_name: str) -> Callable[[torch.device], GenomeEmbedding]:
    if model_name.startswith("evo-1"):
        tokens = model_name.split("_hyena")
        if len(tokens) == 1:
            evo_checkpoint_name = tokens[0]
            num_hyena_layers = 32
        elif len(tokens) == 2:
            evo_checkpoint_name = tokens[0]
            num_hyena_layers = int(tokens[-1])
        else:
            raise RuntimeError("Incorrect model name syntax. Expected '<evo_checkpoint_name>_hyena<n_layers>', but got {} instead.".format(model_name))

        from gem.glms.evo import EvoWrapper
        model_fn = lambda device: EvoWrapper(device=device, num_hyena_layers=num_hyena_layers, checkpoint_name=evo_checkpoint_name)
    elif model_name.startswith("evo2"):
        tokens = model_name.split("_hyena")
        if len(tokens) == 1:
            evo2_checkpoint_name = tokens[0]
            num_hyena_layers = 32
        elif len(tokens) == 2:
            evo2_checkpoint_name = tokens[0]
            num_hyena_layers = int(tokens[-1])
        else:
            raise RuntimeError("Incorrect model name syntax. Expected '<evo2_checkpoint_name>_hyena<n_layers>', but got {} instead.".format(model_name))

        from gem.glms.evo2 import Evo2Wrapper
        model_fn = lambda device: Evo2Wrapper(device=device, num_hyena_layers=num_hyena_layers, checkpoint_name=evo2_checkpoint_name)
    elif model_name == 'dnabert-s':
        from gem.glms.dnabert import DNABertSWrapper
        model_fn = lambda device: DNABertSWrapper(device=device)
    else:
        raise ValueError("Unknown model name {}".format(model_name))
    return model_fn


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asv_fasta_file", type=str, required=True)
    parser.add_argument("--hdf5_output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--embed_batch_size", type=int, required=True)
    parser.add_argument("--cuda_device_ids", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        asv_fasta_file=Path(args.asv_fasta_file),
        hdf5_output_path=Path(args.hdf5_output_path),
        model_name=args.model_name,
        embed_batch_size=args.embed_batch_size,
        cuda_devices=parse_cuda_device_ids(args.cuda_device_ids),
    )
