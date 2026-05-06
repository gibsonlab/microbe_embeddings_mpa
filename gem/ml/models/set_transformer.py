import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
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
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True):
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

    def forward(self, Q, X, mask=None):
        """
        :param Q: Query tensor of shape (B, n, dim_Q).
        :param X: Key/value tensor of shape (B, m, dim_K).
        :param mask: Boolean tensor of shape (B, m), where True indicates a
            real (non-padding) position. Padding positions are masked out of
            the softmax. Default None (no masking).
        :return: Output tensor of shape (B, n, dim_V).
        """
        B, n, _ = Q.shape
        h, d = self.num_heads, self.dim_V // self.num_heads

        def split_heads(t):
            B, s, _ = t.shape
            return t.view(B, s, h, d).transpose(1, 2).reshape(B * h, s, d)

        Q_ = split_heads(self.fc_q(Q))
        K_ = split_heads(self.fc_k(X))
        V_ = split_heads(self.fc_v(X))

        scores = torch.bmm(Q_, K_.transpose(1, 2)) / math.sqrt(d)  # (B*h, n, m)

        if mask is not None:
            # mask: (B, m) -> (B, 1, 1, m) -> (B*h, 1, m) after repeat
            # False (padding) positions become -inf so softmax zeroes them out
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, m)
            mask_expanded = mask_expanded.expand(B, h, n, -1)        # (B, h, n, m)
            mask_expanded = mask_expanded.reshape(B * h, n, -1)      # (B*h, n, m)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        out = torch.bmm(F.softmax(scores, dim=-1), V_)
        out = out.view(B, h, n, d).transpose(1, 2).reshape(B, n, self.dim_V)
        out = self.fc_o(out)

        H = self.ln0(self.fc_q(Q) + out)
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
    """

    def __init__(self, dim_in, dim_out, num_heads, ln=True):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
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
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=True):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in,  dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in,  dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        """
        :param X: Input set tensor of shape (B, n, dim_in).
        :param mask: Boolean tensor of shape (B, n), True for real positions.
            Default None (no masking).
        :return: Output tensor of shape (B, n, dim_out).
        """
        B = X.size(0)
        # mask applied here: inducing points must not attend to padding in X
        H = self.mab0(self.I.expand(B, -1, -1), X, mask=mask)
        # no mask here: H has no padding (it has fixed shape num_inds)
        return self.mab1(X, H)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA) from Lee et al. 2019.

    Pools a set of n vectors into k output vectors using k learned seed vectors S:
        out = MAB(S, rFF(Z))

    Supports a key padding mask so that padding positions in Z do not
    contribute to the pooled output.

    :param dim: Feature dimension for both input and output.
    :param num_heads: Number of attention heads. Must divide dim evenly.
    :param num_seeds: Number of seed vectors k to pool into.
    :param ln: Whether to apply LayerNorm inside MAB. Default True.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=True):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def forward(self, Z, mask=None):
        """
        :param Z: Input set tensor of shape (B, n, dim).
        :param mask: Boolean tensor of shape (B, n), True for real positions.
            Default None (no masking).
        :return: Pooled output tensor of shape (B, num_seeds, dim).
        """
        B = Z.size(0)
        return self.mab(self.S.expand(B, -1, -1), self.ff(Z), mask=mask)


class HierarchicalSetTransformer(nn.Module):
    """
    Hierarchical Set Transformer for inputs of shape (B, N, G, D),
    producing one logit per N element. Apply softmax outside the model.

    Accepts two padding masks:
        - mask_G of shape (B, N, G): True for real G elements, False for padding.
        - mask_N of shape (B, N):    True for real N elements, False for padding.

    mask_G is used during stage 1 so that padded G positions are ignored
    when encoding and pooling each (G, D) set.

    mask_N is used during stage 2 so that padded N positions are ignored
    when encoding and pooling the N summary vectors. Logits at padded N
    positions are set to -inf in the output so they become zero probability
    after an external softmax.

    See __init__ for architecture details.

    :param marker_embed_dim: Input feature dimension D.
    :param dim_hidden: Internal feature dimension used throughout. Default 128.
    :param num_inds: Number of inducing points in each ISAB block. Default 16.
    :param num_heads: Number of attention heads. Must divide dim_hidden evenly.
        Default 4.
    :param ln: Whether to apply LayerNorm inside all attention blocks.
        Default True.
    """

    def __init__(
        self,
        marker_embed_dim,
        dim_hidden=128,
        num_inds=16,
        num_heads=4,
        ln=True,
    ):
        super().__init__()

        self.inner_encoder = nn.Sequential(
            ISAB(marker_embed_dim,  dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.inner_pool = PMA(dim_hidden, num_heads, num_seeds=1, ln=ln)

        self.outer_encoder = nn.Sequential(
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.outer_pool = PMA(dim_hidden, num_heads, num_seeds=1, ln=ln)

        self.decoder_mab = MAB(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln)
        self.output_head = nn.Linear(dim_hidden, 1)

    def forward(self, X, mask_G=None, mask_N=None):
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

        # Flatten mask_G from (B, N, G) to (B*N, G) to match the merged batch dim
        mask_G_flat = mask_G.view(B * N, G) if mask_G is not None else None

        for layer in self.inner_encoder:
            X_inner = layer(X_inner, mask=mask_G_flat)

        X_inner = self.inner_pool(X_inner, mask=mask_G_flat)  # (B*N, 1, dim_hidden)
        N_summaries = X_inner.squeeze(1).view(B, N, -1)       # (B, N, dim_hidden)

        # ── Stage 2: aggregate over N -> global context ───────────────────
        for layer in self.outer_encoder:
            N_summaries = layer(N_summaries, mask=mask_N)

        global_ctx = self.outer_pool(N_summaries, mask=mask_N)  # (B, 1, dim_hidden)

        # ── Decoder: per-N prediction conditioned on global context ───────
        out = self.decoder_mab(N_summaries, global_ctx)   # (B, N, dim_hidden)
        logits = self.output_head(out).squeeze(-1)        # (B, N)

        # Mask padded N positions to -inf so they vanish under external softmax
        if mask_N is not None:
            logits = logits.masked_fill(~mask_N, float('-inf'))

        return logits
