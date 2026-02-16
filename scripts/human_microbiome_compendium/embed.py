from typing import *
from pathlib import *
import argparse

import torch
from Bio import SeqIO

from gem.glms.picker import pick_model_function
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
        multi_alignment_fasta: Optional[Path],
):
    """
    :param asv_fasta_file: Path to a multi-FASTA nucleotide sequence file, where each record is
    the consensus sequence of an ASV.
    :param hdf5_output_path:
    :param model_name:
    :param embed_batch_size:
    :param cuda_devices:
    :param multi_alignment_fasta: Path to the multi-FASTA alignment file, which is the multiple alignment of all
    sequences in asv_fasta_file. Required only if requested model is UMAP.
    :return:
    """
    asv_seqs = load_fasta_dict(asv_fasta_file)
    embed_create_fn, _ = pick_model_function(
        model_name,
        unaligned_fasta=asv_fasta_file,
        multi_alignment_fasta=multi_alignment_fasta
    )

    hdf5_parent_dir = hdf5_output_path.parent
    if not hdf5_parent_dir.exists():
        print("Creating directory: {hdf5_parent_dir}")
        hdf5_parent_dir.mkdir(parents=True)

    if model_name.startswith("umap") or model_name.startswith("pcoa"):
        print("UMAP/PCOA embedding requested: using singular CPU device instead of CUDA.")
        cuda_devices = ['cpu']
    precompute_embeddings(asv_seqs, embed_create_fn, hdf5_output_path, embed_batch_size, cuda_devices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asv_fasta_file", type=str, required=True)
    parser.add_argument("--hdf5_output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--embed_batch_size", type=int, required=True)
    parser.add_argument("--cuda_device_ids", type=str, required=True)
    parser.add_argument("--multi_align_path", type=str, required=False, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if len(args.multi_align_path) == 0:
        multi_align_path = None
    else:
        multi_align_path = Path(args.multi_align_path)

    from gem.util import parse_cuda_device_ids
    main(
        asv_fasta_file=Path(args.asv_fasta_file),
        hdf5_output_path=Path(args.hdf5_output_path),
        model_name=args.model_name,
        embed_batch_size=args.embed_batch_size,
        cuda_devices=parse_cuda_device_ids(args.cuda_device_ids),
        multi_alignment_fasta=multi_align_path,
    )
