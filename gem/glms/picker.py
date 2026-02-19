from typing import Tuple, Callable
from pathlib import Path
import torch
from .base import GenomeEmbedding


InitializerType=Tuple[Callable[[torch.device], GenomeEmbedding], int, torch.dtype]


def evo1_initializer(model_name: str) -> InitializerType:
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
        raise RuntimeError(
            "Incorrect model name syntax. Expected '<evo_checkpoint_name>_hyena<n_layers>', but got {} instead.".format(
                model_name))

    from gem.glms.evo import EvoWrapper
    model_fn = lambda device: EvoWrapper(device=device, num_hyena_layers=num_hyena_layers,
                                         checkpoint_name=evo_checkpoint_name)
    return model_fn, 4096, torch.bfloat16


def evo2_initializer(model_name: str) -> InitializerType:
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
        raise RuntimeError(
            "Incorrect model name syntax. Expected '<evo2_checkpoint_name>_hyena<n_layers>', but got {} instead.".format(
                model_name))

    from gem.glms.evo2 import Evo2Wrapper
    model_fn = lambda device: Evo2Wrapper(device=device, num_hyena_layers=num_hyena_layers,
                                          checkpoint_name=evo2_checkpoint_name)
    return model_fn, 4096, torch.bfloat16


def dnabert_s_initializer():
    from gem.glms.dnabert import DNABertSWrapper
    model_fn = lambda device: DNABertSWrapper(device=device)
    return model_fn, 768, torch.float32


def umap_initializer(model_name: str, **kwargs):
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

    assert 'unaligned_fasta' in kwargs and isinstance(kwargs['unaligned_fasta'],
                                                      Path), "For UMAP embeddings, the `unaligned_fasta` path is required."
    assert 'multi_alignment_fasta' in kwargs and isinstance(kwargs['multi_alignment_fasta'],
                                                            Path), "For UMAP embeddings, the `multi_alignment_fasta` path is required."

    from gem.glms.umap import UMAPEmbedding
    embedding = UMAPEmbedding(
        unaligned_fasta=kwargs['unaligned_fasta'],
        multi_alignment_fasta=kwargs['multi_alignment_fasta'],
        embed_dim=embed_dim,
        rng_seed=rng_seed
    )
    model_fn = lambda _: embedding
    return model_fn, embed_dim, torch.float32


def pcoa_initializer(model_name: str, **kwargs):
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

    assert 'unaligned_fasta' in kwargs and isinstance(kwargs['unaligned_fasta'],
                                                      Path), "For PCoA embeddings, the `unaligned_fasta` path is required."
    assert 'multi_alignment_fasta' in kwargs and isinstance(kwargs['multi_alignment_fasta'],
                                                            Path), "For PCoA embeddings, the `multi_alignment_fasta` path is required."

    from gem.glms.pcoa import PCoAEmbedding
    embedding = PCoAEmbedding(
        unaligned_fasta=kwargs['unaligned_fasta'],
        multi_alignment_fasta=kwargs['multi_alignment_fasta'],
        embed_dim=embed_dim,
        rng_seed=rng_seed,
        chunk_size=20,
    )
    model_fn = lambda _: embedding
    return model_fn, embed_dim, torch.float32


def pick_model_function(model_name: str, **kwargs) -> InitializerType:
    if model_name.startswith("evo-1"):
        return evo1_initializer(model_name)
    elif model_name.startswith("evo2"):
        return evo2_initializer(model_name)
    elif model_name == 'dnabert-s':
        return dnabert_s_initializer()
    elif model_name.startswith("umap"):
        return umap_initializer(model_name, **kwargs)
    elif model_name.startswith("pcoa"):
        return pcoa_initializer(model_name, **kwargs)
    else:
        raise ValueError("Unknown model name {}".format(model_name))
