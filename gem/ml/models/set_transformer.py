import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LinearInitializedModule


class MAB(LinearInitializedModule):
    """
    Multihead Attention Block (MAB) from Lee et al. 2019.

    Computes cross-attention from queries Q over keys/values X:
        H   = LayerNorm(Q + Multihead(Q, X, X))
        out = LayerNorm(H + rFF(H))

    Optionally accepts a key padding mask to ignore padded positions in X.

    :param dim_Q: Input dimension of queries.
    :param dim_K: Input dimension of keys/values.
    :param dim_V: Output dimension (also the internal attention dimension).
    :param num_heads: Number of attention heads. Must divide dim_V evenly.
    :param ln: Whether to apply LayerNorm. Default True.
    :param init_rng: Optional RNG for reproducible parameter initialisation.
    """

    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        num_heads: int,
        ln: bool = True,
        init_rng: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_V = dim_V
        assert dim_V % num_heads == 0

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ff = nn.Sequential(
            nn.Linear(dim_V, dim_V), nn.ReLU(), nn.Linear(dim_V, dim_V)
        )

        if init_rng is not None:
            self.init_weights(init_rng)

    def forward(self, Q: torch.Tensor, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, n, _ = Q.shape
        h, d = self.num_heads, self.dim_V // self.num_heads

        Q_proj = self.fc_q(Q)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            nb, s, _ = t.shape
            return t.view(nb, s, h, d).transpose(1, 2)

        Q_ = split_heads(Q_proj)
        K_ = split_heads(self.fc_k(X))
        V_ = split_heads(self.fc_v(X))

        # Boolean mask: shape (B, 1, 1, m), True = real position (attend to it).
        # SDPA with a bool mask keeps the reduction fused under torch.compile,
        # avoiding the "online softmax disabled" warning from float mask tiling.
        attn_mask = mask[:, None, None, :] if mask is not None else None

        out = F.scaled_dot_product_attention(Q_, K_, V_, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().reshape(B, n, self.dim_V)
        out = self.fc_o(out)

        H = self.ln0(Q_proj + out)
        return self.ln1(H + self.ff(H))


class SAB(nn.Module):
    """
    Set Attention Block (SAB) from Lee et al. 2019.

    Self-attention over a set: SAB(X) = MAB(X, X).
    Complexity is O(n^2) in the set size n.

    :param dim_in: Input feature dimension.
    :param dim_out: Output feature dimension.
    :param num_heads: Number of attention heads. Must divide dim_out evenly.
    :param ln: Whether to apply LayerNorm inside MAB. Default True.
    :param init_rng: Optional RNG for reproducible parameter initialisation.
        Passed through to the internal MAB.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        ln: bool = True,
        init_rng: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, init_rng=init_rng)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param X: Input set tensor of shape (B, n, dim_in).
        :param mask: Boolean tensor of shape (B, n), True for real positions.
            Default None (no masking).
        :return: Output tensor of shape (B, n, dim_out).
        """
        return self.mab(X, X, mask=mask)


class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB) from Lee et al. 2019.

    Approximates SAB using m learned inducing points I, reducing complexity
    from O(n^2) to O(n*m):
        H    = MAB(I, X)   -- inducing points attend over the input set
        out  = MAB(X, H)   -- input set attends over the inducing points

    The mask is applied in the first MAB (I attends over X), so padding in X
    is ignored. The second MAB (X attends over H) needs no mask because H has
    fixed size m with no padding.

    :param dim_in: Input feature dimension.
    :param dim_out: Output feature dimension.
    :param num_heads: Number of attention heads. Must divide dim_out evenly.
    :param num_inds: Number of inducing points m. Typically m << n.
    :param ln: Whether to apply LayerNorm inside MAB. Default True.
    :param init_rng: Optional RNG for reproducible parameter initialisation.
        Used to initialise the inducing point tensor I and passed through to
        both internal MAB modules.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        num_inds: int,
        ln: bool = True,
        init_rng: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I.data, generator=init_rng)

        self.mab0 = MAB(dim_out, dim_in,  dim_out, num_heads, ln=ln, init_rng=init_rng)
        self.mab1 = MAB(dim_in,  dim_out, dim_out, num_heads, ln=ln, init_rng=init_rng)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param X: Input set tensor of shape (B, n, dim_in).
        :param mask: Boolean tensor of shape (B, n), True for real positions.
            Default None (no masking).
        :return: Output tensor of shape (B, n, dim_out).
        """
        B = X.size(0)
        H = self.mab0(self.I.expand(B, -1, -1), X, mask=mask)
        return self.mab1(X, H)


class PMA(LinearInitializedModule):
    """
    Pooling by Multihead Attention (PMA) from Lee et al. 2019.

    Pools a set of n vectors into k output vectors using k learned seed vectors S:
        out = MAB(S, rFF(Z))

    Supports a key padding mask so that padding positions in Z do not
    contribute to the pooled output. Padding positions are zeroed before
    being passed through rFF to prevent large arbitrary activations from
    destabilising LayerNorm, even though the MAB mask would otherwise
    suppress their contribution to the attention output.

    :param dim: Feature dimension for both input and output.
    :param num_heads: Number of attention heads. Must divide dim evenly.
    :param num_seeds: Number of seed vectors k to pool into.
    :param ln: Whether to apply LayerNorm inside MAB. Default True.
    :param init_rng: Optional RNG for reproducible parameter initialisation.
        Used to initialise the seed vector tensor S and the direct ff layers,
        and passed through to the internal MAB.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int,
        ln: bool = True,
        init_rng: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S.data, generator=init_rng)

        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, init_rng=init_rng)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        if init_rng is not None:
            self.init_weights(init_rng)

    def forward(self, Z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param Z: Input set tensor of shape (B, n, dim).
        :param mask: Boolean tensor of shape (B, n), True for real positions.
            Default None (no masking).
        :return: Pooled output tensor of shape (B, num_seeds, dim).
        """
        B = Z.size(0)

        # Zero out padding positions before rFF to prevent large activations
        # from propagating into the attention values, even though they would
        # be masked in MAB. This is a purely defensive measure.
        if mask is not None:
            Z = Z * mask.unsqueeze(-1).float()

        return self.mab(self.S.expand(B, -1, -1), self.ff(Z), mask=mask)


class HierarchicalSetTransformer(LinearInitializedModule):
    """
    Hierarchical Set Transformer for inputs of shape (B, N, G, D),
    producing one logit per N element. Apply softmax outside the model.

    The forward pass has two encoder stages followed by a self-attention decoder:

    - Stage 1 (aggregate over G): each of the N elements has G sub-elements.
      These are encoded independently via an ISAB stack and pooled with PMA
      (num_seeds=1) to produce one summary vector per (sample, N-element).
      Shape: (B*N, G, D) -> (B, N, dim_hidden).

    - Stage 2 (encode over N): the N summary vectors are encoded via a
      second ISAB stack so that each summary absorbs information about the
      rest of the set.
      Shape: (B, N, dim_hidden) -> (B, N, dim_hidden).

    - Decoder: a final self-attention block (MAB(X, X)) lets each N summary
      attend over the full set of N summaries, producing per-element
      representations that are conditioned on every other element. A linear
      head maps each attended vector to a scalar logit.
      Shape: (B, N, dim_hidden) -> (B, N).

    The model is permutation-invariant over G and permutation-equivariant over N.

    Accepts two padding masks:
        - mask_G of shape (B, N, G): True for real G elements, False for padding.
        - mask_N of shape (B, N):    True for real N elements, False for padding.

    mask_G is used during stage 1 so that padded G positions are ignored
    when encoding and pooling each (G, D) set.

    mask_N is used during stage 2 and in the decoder so that padded N
    positions are ignored. Logits at padded N positions are set to -inf in
    the output so they become zero probability after an external softmax.

    :param marker_embed_dim: Input feature dimension D.
    :param dim_hidden: Internal feature dimension used throughout. Default 128.
    :param num_inds: Number of inducing points in each ISAB block. Default 16.
    :param num_heads: Number of attention heads. Must divide dim_hidden evenly.
        Default 4.
    :param ln: Whether to apply LayerNorm inside all attention blocks.
        Default True.
    :param init_rng: Optional RNG for reproducible parameter initialisation.
        Passed down to all child modules.
    """

    def __init__(
        self,
        marker_embed_dim: int,
        dim_hidden: int = 128,
        num_inds: int = 16,
        num_heads: int = 4,
        ln: bool = True,
        init_rng: Optional[torch.Generator] = None,
    ):
        super().__init__()

        self.inner_encoder = nn.ModuleList([
            ISAB(marker_embed_dim, dim_hidden, num_heads, num_inds, ln=ln, init_rng=init_rng),
            ISAB(dim_hidden,       dim_hidden, num_heads, num_inds, ln=ln, init_rng=init_rng),
            ISAB(dim_hidden,       dim_hidden, num_heads, num_inds, ln=ln, init_rng=init_rng),
        ])
        self.inner_pool = PMA(dim_hidden, num_heads, num_seeds=1, ln=ln, init_rng=init_rng)

        self.outer_encoder = nn.ModuleList([
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, init_rng=init_rng),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, init_rng=init_rng),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, init_rng=init_rng),
        ])

        # Final self-attention decoder: each N element attends over all N elements.
        self.decoder_sab = SAB(dim_hidden, dim_hidden, num_heads, ln=ln, init_rng=init_rng)
        self.output_head = nn.Linear(dim_hidden, 1)

        if init_rng is not None:
            self.init_weights(init_rng)

    def forward(
        self,
        X: torch.Tensor,
        mask_G: Optional[torch.Tensor] = None,
        mask_N: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param X: Input tensor of shape (B, N, G, D).
        :param mask_G: Boolean tensor of shape (B, N, G), True for real G
            elements. If None, all G positions are treated as real.
        :param mask_N: Boolean tensor of shape (B, N), True for real N
            elements. If None, all N positions are treated as real.
        :return: Logit tensor of shape (B, N). Padded N positions are set to
            -inf. Apply softmax externally.
        """
        B, N, G, D = X.shape

        # ── Stage 1: aggregate over G ─────────────────────────────────────
        X_inner = X.view(B * N, G, D)
        mask_G_flat = mask_G.view(B * N, G) if mask_G is not None else None

        for layer in self.inner_encoder:
            X_inner = layer(X_inner, mask=mask_G_flat)

        X_inner = self.inner_pool(X_inner, mask=mask_G_flat)         # (B*N, 1, dim_hidden)
        N_summaries = X_inner.squeeze(1).view(B, N, -1)              # (B, N, dim_hidden)

        # ── Stage 2: encode over N ────────────────────────────────────────
        for layer in self.outer_encoder:
            N_summaries = layer(N_summaries, mask=mask_N)            # (B, N, dim_hidden)

        # ── Decoder: per-N self-attention ─────────────────────────────────
        out = self.decoder_sab(N_summaries, mask=mask_N)             # (B, N, dim_hidden)
        logits = self.output_head(out).squeeze(-1)                   # (B, N)

        if mask_N is not None:
            logits = logits.masked_fill(~mask_N, float('-inf'))

        return logits