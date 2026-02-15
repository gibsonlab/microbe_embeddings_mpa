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
    embed_create_fn = embedding_model_initializer(
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


def embedding_model_initializer(model_name: str, **kwargs) -> Callable[[torch.device], GenomeEmbedding]:
    if model_name.startswith("evo-1"):
        """
        format is: <evo1_checkpoint_name>_hyena<n_layers>
        example: evo-1-8k-base_hyena5 is the "evo-1-8k-base" checkpoint, using the first 5 hyena layers.
        """
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
        """
        Format is: <evo2_checkpoint_name>_hyena<n_layers>
        example: evo2_7b_hyena10 is the "evo2_7b" checkpoint, using the first 10 hyena layers.
        """
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
    elif model_name.startswith("umap"):
        """ 
        Format is: umap_d<dims>_s<seed> 
        example: umap_d20_s1234 is UMAP trained to output d=20 embeddings, initialized with seed 1234.
        """
        error_msg = "Incorrect model name syntax. Expected umap_d<dims>_s<seed>, but got {} instead.".format(model_name)

        tokens = model_name.split("_")
        assert len(tokens) == 3, error_msg
        umap_str, dim_str, seed_str = tokens
        assert umap_str == "umap" and dim_str.startswith("d") and seed_str.startswith("s"), error_msg
        try:
            embed_dim = int(dim_str[1:])
            rng_seed = int(seed_str[1:])
        except ValueError:
            raise RuntimeError(error_msg) from None

        assert 'unaligned_fasta' in kwargs and isinstance(kwargs['unaligned_fasta'], Path), "For UMAP embeddings, the `unaligned_fasta` path is required."
        assert 'multi_alignment_fasta' in kwargs and isinstance(kwargs['multi_alignment_fasta'], Path), "For UMAP embeddings, the `multi_alignment_fasta` path is required."

        from gem.glms.umap import UMAPEmbedding
        embedding = UMAPEmbedding(
            unaligned_fasta=kwargs['unaligned_fasta'],
            multi_alignment_fasta=kwargs['multi_alignment_fasta'],
            embed_dim=embed_dim,
            rng_seed=rng_seed
        )
        model_fn = lambda _: embedding
    elif model_name.startswith("pcoa"):
        """ 
        Format is: pcoa_d<dims>_s<seed> 
        example: pcoa_d20_s1234 is PCoA trained to output d=20 embeddings, using seed 1234.
        """
        error_msg = "Incorrect model name syntax. Expected pcoa_d<dims>_s<seed>, but got {} instead.".format(model_name)

        tokens = model_name.split("_")
        assert len(tokens) == 3, error_msg
        umap_str, dim_str, seed_str = tokens
        assert umap_str == "pcoa" and dim_str.startswith("d") and seed_str.startswith("s"), error_msg
        try:
            embed_dim = int(dim_str[1:])
            rng_seed = int(seed_str[1:])
        except ValueError:
            raise RuntimeError(error_msg) from None

        assert 'unaligned_fasta' in kwargs and isinstance(kwargs['unaligned_fasta'], Path), "For PCoA embeddings, the `unaligned_fasta` path is required."
        assert 'multi_alignment_fasta' in kwargs and isinstance(kwargs['multi_alignment_fasta'], Path), "For PCoA embeddings, the `multi_alignment_fasta` path is required."

        from gem.glms.pcoa import PCoAEmbedding
        embedding = PCoAEmbedding(
            unaligned_fasta=kwargs['unaligned_fasta'],
            multi_alignment_fasta=kwargs['multi_alignment_fasta'],
            embed_dim=embed_dim,
            rng_seed=rng_seed
        )
        model_fn = lambda _: embedding
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
    parser.add_argument("--multi_align_path", type=str, required=False, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if len(args.multi_align_path) == 0:
        multi_align_path = None
    else:
        multi_align_path = Path(args.multi_align_path)

    main(
        asv_fasta_file=Path(args.asv_fasta_file),
        hdf5_output_path=Path(args.hdf5_output_path),
        model_name=args.model_name,
        embed_batch_size=args.embed_batch_size,
        cuda_devices=parse_cuda_device_ids(args.cuda_device_ids),
        multi_alignment_fasta=multi_align_path,
    )
