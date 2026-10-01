import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _check_mask(mask: Tensor, scores: Tensor) -> None:
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool (True = may attend), got {mask.dtype}")
    if mask.dim() != 4:
        raise ValueError(
            f"mask must be 4-D (B|1, 1, T|1, S), got {tuple(mask.shape)}; build it with "
            "model.masks.attention_mask rather than relying on broadcasting"
        )
    b, _, t, s = scores.shape
    mb, mh, mt, ms = mask.shape
    if mb not in (1, b) or mh != 1 or mt not in (1, t) or ms != s:
        raise ValueError(f"mask {tuple(mask.shape)} does not fit scores {tuple(scores.shape)}")


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """q (B, H, T, Dh), k and v (B, H, S, Dh). Returns output (B, H, T, Dh) and weights (B, H, T, S)."""
    # 1/sqrt(Dh) keeps the score variance near 1 for unit-variance q and k, so softmax does
    # not saturate into one-hot at init and starve every other key of gradient.
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

    if mask is None:
        weights = F.softmax(scores, dim=-1)
    else:
        _check_mask(mask, scores)
        # -inf, not a large finite negative: exp(-inf) is exactly 0.0, so a masked key gets
        # exactly zero weight and exactly zero gradient. A soft fill like -1e4 is also 0.0 in
        # float32 after exp, but -20 or a float16 fill is not, and the leak would be tiny.
        scores = scores.masked_fill(~mask, float("-inf"))
        # A row with no visible key would be softmax over all -inf = nan, and nan survives a
        # later where() because softmax's backward computes nan * 0. So such rows are made
        # finite before the softmax and zeroed after: the query attends to nothing and its
        # output is 0. Masking keys of right-padded sequences never produces such a row on
        # its own; a fully padded sequence or a left-padded causal row does.
        has_key = mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~has_key, 0.0)
        weights = F.softmax(scores, dim=-1) * has_key

    if training and dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        # Separate projections rather than one fused (3D, D) matrix, so cross-attention can
        # project queries from the decoder and keys/values from the encoder without slicing.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x: Tensor) -> Tensor:
        # (B, L, D) -> (B, H, L, Dh). The view must split D, then the transpose moves heads
        # out. Viewing straight to (B, H, L, Dh) has the right shape but scatters time steps
        # across "heads" and "positions", which is a causal leak no mask can prevent.
        b, length, _ = x.shape
        return x.view(b, length, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        b, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(b, length, self.d_model)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """query (B, T, D), key and value (B, S, D). Returns (B, T, D) and per-head weights (B, H, T, S)."""
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        out, weights = scaled_dot_product_attention(
            q, k, v, mask, dropout_p=self.dropout, training=self.training
        )
        return self.out_proj(self._merge(out)), weights
