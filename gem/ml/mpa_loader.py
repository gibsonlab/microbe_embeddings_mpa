from typing import *

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from gem.mpa import MetaphlanDataset


def collate_sample_batch(
        batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Custom collate function for microbiome data.

    :param batch: List of (features, marker_masks, sgb_masks, targets) tuples
    :return: Batched (features_batch, marker_masks_batch, sgb_masks_batch, targets_batch)
    """
    features_list, marker_mask_list, sgb_mask_list, targets_list = zip(*batch)

    # Stack the tensors
    features_batch = torch.stack(features_list, dim=0)  # (batch_size, max_num_sgbs, max_num_markers, embedding_dim)
    targets_batch = torch.stack(targets_list, dim=0)  # (batch_size, max_num_sgbs)
    marker_masks_batch = torch.stack(marker_mask_list, dim=0)
    sgb_masks_batch = torch.stack(sgb_mask_list, dim=0)

    return features_batch, marker_masks_batch, sgb_masks_batch, targets_batch


class MetaphlanDataLoader(DataLoader):
    def __init__(
            self,
            dataset: MetaphlanDataset,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False,
            **dataloader_kwargs
    ):
        """
        Initialize the microbiome data loader.

        :param dataset: MetaphlanDataset object
        :param batch_size: Batch size
        :param shuffle: Whether to shuffle data
        :param num_workers: Number of worker processes
        :param pin_memory: Whether to pin memory
        :param dataloader_kwargs: Additional DataLoader arguments
        """
        # Create dataset
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_sample_batch,
            **dataloader_kwargs
        )