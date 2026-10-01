import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from model.decoder import Decoder
from model.encoder import Encoder
from model.masks import attention_mask, causal_mask, pad_mask
from model.pos import LearnedPositions, make_positions
from tokenizer.bpe import PAD


@dataclass
class TransformerConfig:
    src_vocab: int
    tgt_vocab: int
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_enc: int = 3
    n_dec: int = 3
    dropout: float = 0.1
    pre_norm: bool = True
    positions: str = "sinusoidal"
    max_len: int = 128
    scale_embeddings: bool = True
    tie_output: bool = True


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.src_embed = nn.Embedding(cfg.src_vocab, d, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab, d, padding_idx=PAD)
        self.positions = make_positions(cfg.positions, d, cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = Encoder(cfg.n_enc, d, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.pre_norm)
        self.decoder = Decoder(cfg.n_dec, d, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.pre_norm)
        self.generator = nn.Linear(d, cfg.tgt_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # std d^-0.5 so that after the sqrt(d) scaling an embedding has unit variance per
        # dimension, the same scale as the sinusoidal signal (values in [-1, 1]). Without the
        # scaling, token identity is ~16x quieter than position at d=256: that is the ablation.
        for table in (self.src_embed, self.tgt_embed):
            nn.init.normal_(table.weight, std=d**-0.5)
            with torch.no_grad():
                table.weight[PAD].zero_()
        if cfg.tie_output:
            self.generator.weight = self.tgt_embed.weight
        if isinstance(self.positions, LearnedPositions):
            nn.init.normal_(self.positions.table.weight, std=0.02)

    def _prepare(self, embedded: Tensor) -> Tensor:
        if self.cfg.scale_embeddings:
            embedded = embedded * math.sqrt(self.cfg.d_model)
        return self.dropout(self.positions(embedded))

    def encode(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """src (B, S) token ids. Returns memory (B, S, D) and the (B, S) source pad mask."""
        src_pad = pad_mask(src, PAD)
        memory = self.encoder(self._prepare(self.src_embed(src)), attention_mask(src_pad))
        return memory, src_pad

    def decode_embedded(
        self, tgt_embedded: Tensor, memory: Tensor, src_pad: Tensor, tgt_pad: Tensor | None
    ) -> tuple[Tensor, list[Tensor]]:
        """Decoder from raw target embeddings (B, T, D), before scaling and positions, so the
        leakage test can differentiate the full path with respect to a continuous input."""
        self_mask = attention_mask(tgt_pad, causal_mask(tgt_embedded.size(1), tgt_embedded.device))
        hidden, cross = self.decoder(self._prepare(tgt_embedded), memory, self_mask, attention_mask(src_pad))
        return self.generator(hidden), cross

    def decode(self, tgt: Tensor, memory: Tensor, src_pad: Tensor) -> tuple[Tensor, list[Tensor]]:
        return self.decode_embedded(self.tgt_embed(tgt), memory, src_pad, pad_mask(tgt, PAD))

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        """Teacher-forced logits (B, T, V): tgt_in starts with BOS, position t predicts token t+1."""
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)[0]
