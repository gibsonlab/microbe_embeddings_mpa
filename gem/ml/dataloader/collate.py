from typing import *
import torch
from torch import Tensor


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
        import os, traceback  # debug
        print(f"Worker PID {os.getpid()} got batch of {len(batch)} samples")  # debug

        # Lazy initialization ensures each worker gets its own buffer
        self._init_buffer()

        # Copy data into pre-allocated buffer.
        # Note: each sample (the i-th element in the batch) may have: n_sgbs <= self.max_num_sgbs, n_markers <= self.max_num_markers.
        feat_buf = self.buffers['features']
        marker_mask_buf = self.buffers['marker_masks']
        sgb_mask_buf = self.buffers['sgb_masks']
        target_buf = self.buffers['targets']

        marker_mask_buf.zero_()
        sgb_mask_buf.zero_()
        target_buf.zero_()

        max_num_sgb = 0
        max_num_markers = 0
        sample_ids = []
        for i, (sample_id, features, marker_mask, sgb_mask, targets) in enumerate(batch):
            feat_buf[i, :features.shape[0], :features.shape[1], :] = features
            marker_mask_buf[i, :marker_mask.shape[0], :marker_mask.shape[1]] = marker_mask
            sgb_mask_buf[i, :sgb_mask.shape[0]] = sgb_mask
            target_buf[i, :targets.shape[0]] = targets

            max_num_sgb = max(max_num_sgb, len(targets))
            max_num_markers = max(max_num_markers, marker_mask.shape[1])
            sample_ids.append(sample_id)

        # Return a slice/copy of the buffer for this batch
        return (
            sample_ids,
            feat_buf[:len(batch), :max_num_sgb, :max_num_markers, :],
            marker_mask_buf[:len(batch), :max_num_sgb, :max_num_markers],
            sgb_mask_buf[:len(batch), :max_num_sgb],
            target_buf[:len(batch), :max_num_sgb]
        )