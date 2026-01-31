from typing import *

import torch
from torch.utils.data import DataLoader

from gem.glms import GenomeEmbedding
from gem.datasets.dataset_sequence import Sample, OrganismDataset

GenomeEmbedding_Subclass = TypeVar("GenomeEmbedding_Subclass", bound=GenomeEmbedding)


class MultiGPUEmbeddingCollateFn:
    def __init__(
            self,
            embedding_class: Type[GenomeEmbedding_Subclass],
            embedding_kwargs: Dict[str, Any],
            device_array: List[Union[str, torch.device]],
            model_batch_size=512,
    ):
        """
        :param embedding_class: The class or factory function to create GenomeEmbedding
        :param model_batch_size: Batch size for model inference
        :param device_array: List of cuda devices to use.
        """
        self.embedding_class = embedding_class
        self.embedding_kwargs = embedding_kwargs
        self.model_batch_size = model_batch_size
        self.device_array = device_array

        # Will be initialized per worker
        self.device = None
        self.worker_id = None
        self.embedding = None
        self.embed_dim = None

    def _init_worker(self):
        """
        Initialize embedding model for this worker on the correct GPU.
        """
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading (num_workers=0)
            self.worker_id = 0
            self.device = self.device_array[0]
        else:
            # Multi-process data loading
            self.worker_id = worker_info.id
            self.device = self.device_array[worker_info.id]

        print(f"Worker {self.worker_id} initializing on {self.device}")

        # Initialize the embedding model on this worker's GPU
        self.embedding: GenomeEmbedding = self.embedding_class(device=self.device, **self.embedding_kwargs)
        self.embed_dim = self.embedding.embed_dim()

    def __call__(self, batch: List[Sample]) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (b, S, G, e) on the worker's assigned GPU
        """
        # Lazy initialization - only create model when first called in worker
        if self.embedding is None:
            self._init_worker()

        b = len(batch)

        # Find max dimensions
        S = max(len(sample) for sample in batch)
        G = max(max(len(organism) for organism in sample) if sample else 0
                for sample in batch)

        # Collect all gene strings with positions
        all_genes = []
        positions = []

        for batch_idx, sample in enumerate(batch):
            for organism_idx, organism in enumerate(sample):
                for gene_idx, gene in enumerate(organism):
                    all_genes.append(gene)
                    positions.append((batch_idx, organism_idx, gene_idx))

        # Batch process through model on this worker's GPU
        if all_genes:
            with torch.no_grad():
                embeddings_list = []

                for i in range(0, len(all_genes), self.model_batch_size):
                    batch_genes = all_genes[i:i + self.model_batch_size]
                    batch_embeddings = self.embedding.embed_batch(batch_genes)
                    embeddings_list.append(batch_embeddings)

                all_embeddings = torch.cat(embeddings_list, dim=0)
        else:
            all_embeddings = torch.zeros((0, self.embed_dim), device=self.device)

        # Initialize output tensor on this worker's GPU
        output = torch.zeros((b, S, G, self.embed_dim), device=self.device)

        # Fill in embeddings
        for emb, (batch_idx, organism_idx, gene_idx) in zip(all_embeddings, positions):
            output[batch_idx, organism_idx, gene_idx] = emb

        return output


def create_dataloader_dynamic_embedding(
        dataset: OrganismDataset,
        embedding_class: Type[GenomeEmbedding_Subclass],
        embedding_kwargs: Dict[str, Any],
        worker_devices: List[torch.device],
        model_batch_size: int = 512,
        **dataloader_kwargs
):
    return DataLoader(
        dataset,
        collate_fn=MultiGPUEmbeddingCollateFn(embedding_class, embedding_kwargs, worker_devices, model_batch_size),
        **dataloader_kwargs
    )
