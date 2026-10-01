from torch import Tensor, nn

from model.attention import MultiHeadAttention
from model.layers import FeedForward, Residual


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.attn_block = Residual(d_model, dropout, pre_norm)
        self.ff_block = Residual(d_model, dropout, pre_norm)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x = self.attn_block(x, lambda h: self.self_attn(h, h, h, mask)[0])
        return self.ff_block(x, self.ff)


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout, pre_norm) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)
