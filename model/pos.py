import math

import torch
from torch import Tensor, nn


class SinusoidalPositions(nn.Module):
    """Vaswani et al. (2017), sec. 3.5: sin on even dims, cos on odd, wavelengths 2π to 10000·2π.

    For any offset k, PE(p + k) is a fixed rotation of PE(p), independent of p. That is the
    property that might let it extrapolate past the training lengths; learned positions have
    nothing to say beyond max_len.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        inv_freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * inv_freq)
        pe[:, 1::2] = torch.cos(position * inv_freq)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(1)]


class LearnedPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.table = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.max_len:
            raise ValueError(f"length {x.size(1)} exceeds learned positions ({self.max_len})")
        return x + self.table.weight[: x.size(1)]


class NoPositions(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


def make_positions(kind: str, d_model: int, max_len: int) -> nn.Module:
    if kind == "sinusoidal":
        return SinusoidalPositions(d_model)
    if kind == "learned":
        return LearnedPositions(d_model, max_len)
    if kind == "none":
        return NoPositions()
    raise ValueError(f"unknown positions {kind!r}")
