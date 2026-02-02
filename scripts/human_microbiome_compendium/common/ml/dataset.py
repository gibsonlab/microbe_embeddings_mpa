from typing import Tuple

import torch
from torch import Tensor

from gem.datasets import AbstractMetaphlanPreembeddedDataset


class ASVPreembeddedDataset(AbstractMetaphlanPreembeddedDataset):
    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        pass

    def __len__(self) -> int:
        pass

    def embedding_dtype(self) -> torch.dtype:
        pass

    def max_num_sgbs(self) -> int:
        pass

    def max_num_markers(self) -> int:
        pass

    def embed_feature_dim(self) -> int:
        pass

    def true_abundance_profile(self, idx: int) -> Tensor:
        pass
