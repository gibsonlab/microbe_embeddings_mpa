from typing import *

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from .base import GenomeEmbedding, kwargs_torch_convert_device


class DNABertSWrapper(GenomeEmbedding):
    def __init__(
            self,
            device: str = 'cuda:0',
    ):
        """ Constructor for the wrapper. """
        """ 
        Note: the official "DNABERT-*" model hosted by author zhihan1996 won't work in newer environments, due to its dependency on an older version of triton package (triton-2.0.0). 
        Evo/Hyena uses newer FlashAttention+Triton, so for this model we must use a version without the older FlashAttention.
        """
        tokenizer = AutoTokenizer.from_pretrained("quietflamingo/dnaberts-no-flashattention", trust_remote_code=True)
        model = AutoModel.from_pretrained("quietflamingo/dnaberts-no-flashattention", trust_remote_code=True)

        print("[DNABert] Special tokens:")
        for name, token in tokenizer.special_tokens_map.items():
            print(f"  {name}: {token} (ID: {tokenizer.convert_tokens_to_ids(token)})")

        if device.startswith("cuda"):
            assert torch.cuda.is_available(), "CUDA is unavailable!"
            torch.cuda.empty_cache()

        model.to(device)
        model.eval()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

        # self.pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)  # added whenever padding is required.
        # self.sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)  # added at the end of each sequence.
    def device(self) -> torch.device:
        return self.device

    def embed_dim(self) -> int:
        # hard-coded!
        return 768

    def embed_sequence(self, x: str) -> Tensor:
        """
        Note: the DNABert-S tokenizer automatically adds a "CLS" classification token at the start.
        This means that after running the model, the first slice contains the info needed for downstream classification tasks from the entire sequence input.
        """
        model_input = self.tokenizer(x, return_tensors='pt')
        with torch.no_grad():
            model_output, _ = self.model(
                **kwargs_torch_convert_device(model_input, self.device))  # shape (1, n_tokens, embed_dim=768)
            return model_output[0, 0]  # output of shape 768, obtained by taking the first slice (see note above)

    def embed_batch(self, seqs: List[str]) -> Tensor:
        model_input = self.tokenizer(seqs, return_tensors='pt', padding=True)
        with torch.no_grad():
            model_output, _ = self.model(
                **kwargs_torch_convert_device(model_input, self.device)
            )  # shape (n_seqs, n_tokens, embed_dim=768). Note that **model_input handles the padding masks automatically.
            return model_output[
                :, 0, :]  # output of shape (n_seqs, 768), obtained by taking the first slice for each seq (see note above)

    def embed_empty_sequence(self) -> Tensor:
        """
        Embed the empty sequence "", which auto-tokenizers to [CLS] [SEP].
        """
        return self.embed_sequence("")