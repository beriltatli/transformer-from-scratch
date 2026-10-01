from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.inner = nn.Linear(d_model, d_ff)
        self.outer = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.outer(self.dropout(F.relu(self.inner(x))))


class Residual(nn.Module):
    """Residual connection around a sublayer, with the norm before or after it.

    post-LN (Vaswani et al.): norm(x + f(x)). Every residual path passes through a norm, so the
    gradient at the bottom layers depends on all the norms above; at a full learning rate from
    step one this is what diverges, hence the warmup.
    pre-LN: x + f(norm(x)). An identity path runs from output to input, so it trains without
    warmup, but the stream is never normalised and the stack needs one final norm.
    """

    def __init__(self, d_model: int, dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        if self.pre_norm:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))
