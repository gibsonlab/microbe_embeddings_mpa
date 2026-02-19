from typing import *

import numpy as np
import torch
from torch import Tensor
from evo import Evo

from .base import GenomeEmbedding


class EvoWrapper(GenomeEmbedding):
    def __init__(
            self,
            num_hyena_layers: int,
            device: torch.device,
            checkpoint_name: str = 'evo-1-8k-base',
    ):
        """
        :param num_hyena_layers: number of hyena layers to use. The final embedding output is the output of the k-th layer (k = num_hyena_layers).
        Note: evo1 pre-trained model is exactly 32 hyena layers.
        :param device: device to use
        """
        print(f"Using Evo checkpoint '{checkpoint_name}'")

        if device.type == 'cuda':
            assert torch.cuda.is_available(), "CUDA is unavailable!"
            torch.cuda.empty_cache()

        evo_model = Evo(checkpoint_name, device=device)
        hyena_model, tokenizer = evo_model.model, evo_model.tokenizer

        ### not needed, StripedHyena already in bfloat16 mode for weights.
        # if half_precision:
        #     # Use half precision (13GB instead of 26GB)
        #     print("using half precision")
        #     model = model.half()

        self.device = device

        if num_hyena_layers > len(hyena_model.blocks):
            raise Exception("Evo model has {} hyena blocks, can't specify num_hyena_layers={}".format(
                len(hyena_model.blocks),
                num_hyena_layers
            ))

        n_total_layers = len(hyena_model.blocks)
        print("[evo] Loaded Hyena Model which has {} blocks. Embedding is output of block #{}".format(n_total_layers, num_hyena_layers))

        # The actual Evo model.
        self.preembedding_layer = hyena_model.embedding_layer
        self.hyena_layers = hyena_model.blocks[:num_hyena_layers]
        self.tokenizer = tokenizer

        for model_component in [self.preembedding_layer] + list(self.hyena_layers):
            # model_component.to(device)
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

    def tokenize_single(self, sequence: str, max_seq_length: Optional[int] = None) -> List[int]:
        if len(sequence) == 0:
            raise ValueError("tokenizing an empty sequence is not allowed.")
        tokenized_ids = self.tokenizer.tokenize(sequence)
        if max_seq_length is not None:
            tokenized_ids += [self.tokenizer.pad_id] * (max_seq_length - len(sequence))
        return tokenized_ids

    def run_hyena(self, input_ids: Tensor) -> Tensor:
        with torch.no_grad():
            x = self.preembedding_layer.embed(input_ids)
            for _, block in enumerate(self.hyena_layers):
                """
                Note: padding_mask is not required here; the evo model is autoregressive.
                This means that the embedding of ("A") is always the prefix of the embedding ("AC"). 
                Equal up to bfloat16 precision, of course.
                """
                x, _ = block(x, inference_params=None, padding_mask=None)
            return x

    def embed_sequence(self, nucleotides: str) -> Tensor:
        input_ids = self.tokenize_single(nucleotides)
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        input_ids = input_ids.unsqueeze(0).to(self.device)
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