from typing import *

import torch
from sklearn.utils import assert_all_finite
from torch import Tensor
from torch.utils.data import DataLoader

from gem.glms import GenomeEmbedding
from gem.datasets.generic import OrganismGeneSequenceDataset
from gem.datasets.generic.types import Sample

GenomeEmbedding_Subclass = TypeVar("GenomeEmbedding_Subclass", bound=GenomeEmbedding)


class MultiGPUEmbeddingCollateFn:
    FALLBACK_DEVICE = torch.device("cuda")
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
        print("Worker device array: {}".format(
            ",".join(str(dev) for dev in device_array)
        ))

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
            self.device = self.FALLBACK_DEVICE
            print(f"No workers being used for embeddings. Using fallback device {self.device} for embedding model {self.embedding_class.__name__}")
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            self.device = self.device_array[worker_id]
            print(f"Worker {worker_id} initializing on {self.device} with embedding model {self.embedding_class.__name__}")

        # Initialize the embedding model on this worker's GPU
        if 'device' in self.embedding_kwargs:
            print("[WARNING] embedding_kwargs contains 'device' kwarg, but this should be automatically specified per worker.")
            del self.embedding_kwargs['device']
        self.embedding: GenomeEmbedding = self.embedding_class(device=self.device, **self.embedding_kwargs)
        self.embed_dim = self.embedding.embed_dim()

    def __call__(self, batch: List[Tuple[Sample, Tensor]]) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
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
        G = max(
            max(len(taxa_genes) for taxa_genes in sample)
            if sample else 0
            for sample in batch
        )

        # things to output
        sample_ids = []
        abundance_tensors = []
        m_batch = torch.zeros((b, S, G), dtype=torch.bool)  # this doesn't need to be created on the worker device.
        s_batch = torch.zeros((b, S), dtype=torch.bool)  # this doesn't need to be created on the worker device.

        # Collect all gene strings with positions
        all_genes = []
        positions = []

        for sample_idx_in_batch, ((sample_id, sample_taxa), abundance_targets) in enumerate(batch):
            sample_ids.append(sample_id)
            abundance_tensors.append(abundance_targets)
            s_batch[sample_idx_in_batch, :len(sample_taxa)] = True
            for taxa_idx, taxa_genes in enumerate(sample_taxa):
                m_batch[sample_idx_in_batch, taxa_idx, :len(taxa_genes)] = True
                for gene_idx, gene in enumerate(taxa_genes):
                    all_genes.append(gene)
                    positions.append((sample_idx_in_batch, taxa_idx, gene_idx))

        # Batch process through model on this worker's GPU
        if len(all_genes) > 0:
            with torch.no_grad():
                embeddings_list = []

                for i in range(0, len(all_genes), self.model_batch_size):
                    minibatch_genes = all_genes[i:i + self.model_batch_size]
                    minibatch_embeddings = self.embedding.embed_batch(minibatch_genes)
                    embeddings_list.append(minibatch_embeddings)

                embedded_genes_flattened = torch.cat(embeddings_list, dim=0)
        else:
            embedded_genes_flattened = torch.zeros((0, self.embed_dim), device=self.device)

        # Initialize output tensor on this worker's GPU, and fill in embeddings in the correct order.
        f_batch = torch.zeros((b, S, G, self.embed_dim), device=self.device)
        for emb, (sample_idx_in_batch, taxa_idx, gene_idx) in zip(embedded_genes_flattened, positions):
            f_batch[sample_idx_in_batch, taxa_idx, gene_idx] = emb

        t_batch = torch.stack(abundance_tensors, dim=0)
        return sample_ids, f_batch, m_batch, s_batch, t_batch


def create_dataloader_dynamic_embedding(
        dataset: OrganismGeneSequenceDataset,
        embedding_class: Type[GenomeEmbedding_Subclass],
        embedding_kwargs: Dict[str, Any],
        worker_devices: List[torch.device],
        data_batch_size: int,
        model_batch_size: int = 512,  # the batch size for pretrained embeddings
        **dataloader_kwargs
):
    return DataLoader(
        dataset,
        collate_fn=MultiGPUEmbeddingCollateFn(embedding_class, embedding_kwargs, worker_devices, model_batch_size),
        batch_size=data_batch_size,
        num_workers=len(worker_devices),
        persistent_workers=True,
        multiprocessing_context='spawn',
        **dataloader_kwargs
    )
