from torch.utils.data import DataLoader

from gem.mpa import AbstractMetaphlanDataset
from .collate import BufferedCollator


def worker_init_fn(worker_id: int, base_rng_seed: int):
    """ Set random seed """
    import random, torch, numpy as np
    worker_seed = base_rng_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class MetaphlanDataLoader(DataLoader):
    def __init__(
            self,
            dataset: AbstractMetaphlanDataset,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False,
            worker_rng_seed: int = 31415,
            **dataloader_kwargs
    ):
        """
        Initialize the microbiome data loader.

        :param dataset: AbstractMetaphlanDataset object
        :param batch_size: Batch size
        :param shuffle: Whether to shuffle data
        :param num_workers: Number of worker processes
        :param pin_memory: Whether to pin memory
        :param dataloader_kwargs: Additional DataLoader arguments
        """
        self.collator = BufferedCollator(
            batch_size=batch_size,
            max_num_sgbs=dataset.max_num_sgbs(),
            max_markers=dataset.max_num_markers(),
            embed_feature_dim=dataset.embed_feature_dim(),
            dtype=dataset.embedding_dtype()
        )
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collator,
            worker_init_fn=lambda wid: worker_init_fn(wid, worker_rng_seed),
            **dataloader_kwargs
        )