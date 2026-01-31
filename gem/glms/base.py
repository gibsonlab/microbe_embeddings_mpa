from abc import abstractmethod, ABC
from typing import *

import torch
from torch import Tensor


""" Generic embedding wrapper class """
class GenomeEmbedding(ABC):
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError()

    @abstractmethod
    def embed_dim(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def embed_sequence(self, x: str) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def embed_batch(self, strs: List[str]) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def embed_empty_sequence(self) -> Tensor:
        raise NotImplementedError()


def kwargs_torch_convert_device(kwargs_dict: Dict, device: str) -> Dict:
    """ In a kwargs dict, transfer any tensors to the specified device. """
    dict_copy = dict()
    for k, v in kwargs_dict.items():
        if isinstance(v, Tensor):
            v = v.to(device)
        dict_copy[k] = v
    return dict_copy
