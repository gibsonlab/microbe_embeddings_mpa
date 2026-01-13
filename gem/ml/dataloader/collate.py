from typing import *
import torch
from torch import Tensor

from gem.util import timer


class BufferedCollator:
    """Collator with pre-allocated buffers per worker."""

    def __init__(
            self,
            batch_size: int,
            max_num_sgbs: int,
            max_markers: int,
            embed_feature_dim: int,
            dtype: torch.dtype,
    ):
        self.batch_size = batch_size
        self.max_num_sgbs = max_num_sgbs
        self.max_markers = max_markers
        self.embed_feature_dim = embed_feature_dim
        self.dtype = dtype

        # Buffers will be created lazily per worker
        self.buffers: Dict[str, Tensor] = dict()

    def _init_buffer(self):
        """Initialize buffer for this worker process."""
        if len(self.buffers) == 0:
            self.buffers['features'] = torch.empty(
                (self.batch_size, self.max_num_sgbs, self.max_markers, self.embed_feature_dim),
                dtype=self.dtype,
                device='cpu',
            )
            self.buffers['marker_masks'] = torch.empty(
                (self.batch_size, self.max_num_sgbs, self.max_markers),
                dtype=torch.bool,
                device='cpu',
            )
            self.buffers['sgb_masks'] = torch.empty(
                (self.batch_size, self.max_num_sgbs),
                dtype=torch.bool,
                device='cpu',
            )
            self.buffers['targets'] = torch.empty(
                (self.batch_size, self.max_num_sgbs),
                dtype=self.dtype,
                device='cpu',
            )

    def __call__(
            self,
            batch: List[Tuple[str, Tensor, Tensor, Tensor, Tensor]]
    ) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
        # import os  # debug
        # print(f"Worker PID {os.getpid()} got batch of {len(batch)} samples")  # debug

        # Lazy initialization ensures each worker gets its own buffer
        self._init_buffer()

        batch_len = len(batch)
        assert batch_len <= self.batch_size, f"Got batch size {batch_len} > buffer batch size {self.batch_size}"

        # Copy data into pre-allocated buffer.
        # Note: each sample (the i-th element in the batch) may have: n_sgbs <= self.max_num_sgbs, n_markers <= self.max_num_markers.
        feat_buf = self.buffers['features']
        marker_mask_buf = self.buffers['marker_masks']
        sgb_mask_buf = self.buffers['sgb_masks']
        target_buf = self.buffers['targets']

        marker_mask_buf[:batch_len].zero_()
        sgb_mask_buf[:batch_len].zero_()
        target_buf[:batch_len].zero_()

        max_num_sgb = 0
        max_num_markers = 0
        sample_ids = []
        for i, (sample_id, features, marker_mask, sgb_mask, targets) in enumerate(batch):
            assert features.shape[0] <= self.max_num_sgbs, f"Sample {i}: sgbs={features.shape[0]}"
            assert features.shape[1] <= self.max_markers, f"Sample {i}: markers={features.shape[1]}"

            feat_buf[i, :features.shape[0], :features.shape[1], :] = features
            marker_mask_buf[i, :marker_mask.shape[0], :marker_mask.shape[1]] = marker_mask
            sgb_mask_buf[i, :sgb_mask.shape[0]] = sgb_mask
            target_buf[i, :targets.shape[0]] = targets

            assert features.shape[0] == len(targets), f"Sample {i}: features.shape[0] = {features.shape[0]}, len(targets) = {len(targets)}"
            max_num_sgb = max(max_num_sgb, features.shape[0])
            max_num_markers = max(max_num_markers, marker_mask.shape[1])
            sample_ids.append(sample_id)

        # Return a slice/copy of the buffer for this batch
        return (
            sample_ids,
            feat_buf[:batch_len, :max_num_sgb, :max_num_markers, :].clone(),
            marker_mask_buf[:batch_len, :max_num_sgb, :max_num_markers].clone(),
            sgb_mask_buf[:batch_len, :max_num_sgb].clone(),
            target_buf[:batch_len, :max_num_sgb].clone()
        )


def collate_fn_stack(
        batch: List[Tuple[str, Tensor, Tensor, Tensor, Tensor]]
) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
    """
    Assuming all samples are pre-padded to same size, just stack.
    This function fails with an error if this size condition is not met.
    """
    sample_ids = [item[0] for item in batch]
    f_batch = torch.stack([item[1] for item in batch], dim=0)
    m_batch = torch.stack([item[2] for item in batch], dim=0)
    s_batch = torch.stack([item[3] for item in batch], dim=0)
    t_batch = torch.stack([item[4] for item in batch], dim=0)
    return sample_ids, f_batch, m_batch, s_batch, t_batch


def collate_fn_dynamic_alloc(
        batch: List[Tuple[str, Tensor, Tensor, Tensor, Tensor]]
) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
    """Minimizes allocations while being multiprocessing-safe"""
    batch_size = len(batch)

    # Find max dimensions
    S_max = max(f.shape[0] for _, f, _, _, _ in batch)
    M_max = max(f.shape[1] for _, f, _, _, _ in batch)
    embed_dim = batch[0][1].shape[-1]

    # Get device and dtype from first sample
    device = batch[0][1].device
    f_dtype = batch[0][1].dtype
    t_dtype = batch[0][4].dtype

    with timer("Collate:Allocate", enabled=True):
        # Single allocation with torch.empty (faster than zeros if you fill everything)
        sample_ids = []
        f_batch = torch.zeros(batch_size, S_max, M_max, embed_dim, dtype=f_dtype, device=device)
        m_batch = torch.zeros(batch_size, S_max, M_max, dtype=torch.bool, device=device)
        s_batch = torch.zeros(batch_size, S_max, dtype=torch.bool, device=device)
        t_batch = torch.zeros(batch_size, S_max, dtype=t_dtype, device=device)

    with timer("Collate:Fill", enabled=True):
        # In-place copy
        for i, (sample_id, f, m, s, t) in enumerate(batch):
            sample_ids.append(sample_id)
            S_i, M_i = f.shape[0], f.shape[1]
            f_batch[i, :S_i, :M_i, :] = f
            m_batch[i, :S_i, :M_i] = m
            s_batch[i, :S_i] = s
            t_batch[i, :S_i] = t

    return sample_ids, f_batch, m_batch, s_batch, t_batch


def fn_no_collation(
        batch: List[Tuple[str, Tensor, Tensor, Tensor, Tensor]]
) -> List[Tuple[str, Tensor, Tensor, Tensor, Tensor]]:
    """Don't pad at all - return list of tensors."""
    return batch