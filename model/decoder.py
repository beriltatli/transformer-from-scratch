from torch import Tensor, nn

from model.attention import MultiHeadAttention
from model.layers import FeedForward, Residual


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.self_block = Residual(d_model, dropout, pre_norm)
        self.cross_block = Residual(d_model, dropout, pre_norm)
        self.ff_block = Residual(d_model, dropout, pre_norm)

    def forward(
        self, x: Tensor, memory: Tensor, self_mask: Tensor | None, cross_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        x = self.self_block(x, lambda h: self.self_attn(h, h, h, self_mask)[0])
        cross_weights = None

        def cross(h: Tensor) -> Tensor:
            nonlocal cross_weights
            out, cross_weights = self.cross_attn(h, memory, memory, cross_mask)
            return out

        x = self.cross_block(x, cross)
        return self.ff_block(x, self.ff), cross_weights


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, n_heads, d_ff, dropout, pre_norm) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, memory: Tensor, self_mask: Tensor | None, cross_mask: Tensor | None
    ) -> tuple[Tensor, list[Tensor]]:
        """Returns hidden states and per-layer cross-attention weights, each (B, H, T, S)."""
        cross = []
        for layer in self.layers:
            x, weights = layer(x, memory, self_mask, cross_mask)
            cross.append(weights)
        return self.final_norm(x), cross
