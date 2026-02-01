from typing import *
from pathlib import Path
import importlib.util
import yaml

import numpy as np
import torch
from torch import Tensor

from .base import GenomeEmbedding


class Evo2Wrapper(GenomeEmbedding):
    """
    Externally, looks the same as EvoWrapper (evo-1 wrapper), but initializes Evo2 instead.
    """
    def __init__(
            self,
            num_hyena_layers: int,
            device: torch.device,
            checkpoint_name: str = 'evo2_7b',
    ):
        """
        :param num_hyena_layers: number of hyena layers to use. The final embedding output is the output of the k-th layer (k = num_hyena_layers).
        Note: evo1 pre-trained model is exactly 32 hyena layers.
        :param device: device to use
        """
        # note: fp8 is disabled directly through the configuration files!!
        # e.g. /usr/local/lib/python3.12/dist-packages/evo2/configs/evo2-7b-8k.yml
        # To turn off fp8, directly modify that configuration file outside of Python.

        print(f"Using Evo2 checkpoint '{checkpoint_name}'")
        from evo2 import Evo2
        evo2_model = Evo2(checkpoint_name)

        hyena_model, tokenizer = evo2_model.model, evo2_model.tokenizer

        ### not needed, StripedHyena already in bfloat16 mode for weights.
        # if half_precision:
        #     # Use half precision (13GB instead of 26GB)
        #     print("using half precision")
        #     model = model.half()

        if device.type == 'cuda':
            assert torch.cuda.is_available(), "CUDA is unavailable!"
            torch.cuda.empty_cache()

        self.device = device

        if num_hyena_layers > len(hyena_model.blocks):
            raise Exception("Evo model has {} hyena blocks, can't specify num_hyena_layers={}".format(
                len(hyena_model.blocks),
                num_hyena_layers
            ))

        n_total_layers = len(hyena_model.blocks)
        print("[evo2] Loaded Hyena Model which has {} blocks. Embedding is output of block #{}".format(n_total_layers, num_hyena_layers))

        # The actual Evo model.
        self.preembedding_layer = hyena_model.embedding_layer
        self.hyena_layers = hyena_model.blocks[:num_hyena_layers]
        self.tokenizer = tokenizer

        for model_component in [self.preembedding_layer] + list(self.hyena_layers):
            model_component.to(device)
            model_component.eval()

        if num_hyena_layers < len(hyena_model.blocks):
            print("[evo] Discarding layers #{} onwards.".format(num_hyena_layers + 1))
            for post_layer in hyena_model.blocks[num_hyena_layers:]:
                del post_layer

    def device(self) -> torch.device:
        return self.device

    def embed_dim(self) -> int:
        example = self.embed_empty_sequence()
        return example.shape[-1]

    def tokenize_single(self, sequence: str, max_seq_length: Optional[int] = None) -> Tensor:
        tokenized_ids = self.tokenizer.tokenize(sequence)
        if max_seq_length is not None:
            tokenized_ids += [self.tokenizer.pad_id] * (max_seq_length - len(sequence))
        return tokenized_ids

    def run_hyena(self, input_ids: Tensor) -> Tensor:
        with torch.no_grad():
            # note: StripedHyena2 seems to have changed the VocabParallelEmbedding implementation detail.
            # Now, it extends nn.Module instead of nn.Embedding, and does not implement the embed() method.
            x = self.preembedding_layer.forward(input_ids)
            for _, block in enumerate(self.hyena_layers):
                """
                Note: padding_mask is not required here; the evo model is autoregressive.
                This means that the embedding of ("A") is always the prefix of the embedding ("AC"). 
                Equal up to bfloat16 precision, of course.
                """
                x, _ = block(x, inference_params=None, padding_mask=None)
            return x

    def embed_sequence(self, nucleotides: str) -> Tensor:
        input_ids = self.tokenize_single(nucleotides, max_seq_length=len(nucleotides))
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        input_ids = input_ids.to(self.device).unsqueeze(0)
        seq_len = len(nucleotides)

        # Note: the "-1" here indicates the indexing of the particular slice UP TO the last character of the sequence.
        return self.run_hyena(input_ids)[0, seq_len - 1, :]

    def embed_batch(self, seqs: List[str]) -> Tensor:
        max_seq_length = max(len(seq) for seq in seqs)
        input_ids = torch.tensor(
            [
                self.tokenize_single(x, max_seq_length=max_seq_length)
                for x in seqs
            ],
            dtype=torch.int
        ).to(self.device)
        hyena_output = self.run_hyena(input_ids)

        return torch.stack([
            seq_hyena_output[len(seq) - 1, :]  # take the last token's embedding vector (the index may differ depending on the sequence)
            for seq, seq_hyena_output in zip(seqs, hyena_output)
        ], dim=0)

    def embed_empty_sequence(self) -> Tensor:
        input_ids = torch.tensor([np.uint8(self.tokenizer.eos)], dtype=torch.int)  # length 1 of "EOS" id.
        input_ids = input_ids.to(self.device).unsqueeze(0)
        return self.run_hyena(input_ids)[0, 0, :]