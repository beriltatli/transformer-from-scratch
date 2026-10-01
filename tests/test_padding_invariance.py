import pytest
import torch

from model.attention import MultiHeadAttention
from model.masks import attention_mask, causal_mask
from model.transformer import Transformer, TransformerConfig
from tokenizer.bpe import PAD

D_MODEL, N_HEADS = 32, 4
LENGTHS = [5, 9, 2, 7]
TARGET = 0


def _batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded batch of LENGTHS. Pad positions hold large noise, not zeros, so an unmasked
    pad key changes the output visibly instead of contributing a near-zero value vector."""
    g = torch.Generator().manual_seed(seed)
    max_len = max(LENGTHS)
    x = 10 * torch.randn(len(LENGTHS), max_len, D_MODEL, generator=g)
    real = torch.arange(max_len)[None, :] < torch.tensor(LENGTHS)[:, None]
    return x, real


@pytest.mark.parametrize("causal", [False, True])
def test_self_attention_alone_equals_padded(causal: bool) -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, N_HEADS).eval()
    x, real = _batch(1)
    n = LENGTHS[TARGET]

    alone_x = x[TARGET : TARGET + 1, :n]
    alone_mask = attention_mask(None, causal_mask(n)) if causal else None
    alone = mha(alone_x, alone_x, alone_x, alone_mask)[0]

    batch_mask = attention_mask(real, causal_mask(x.size(1)) if causal else None)
    batched = mha(x, x, x, batch_mask)[0]

    torch.testing.assert_close(batched[TARGET, :n], alone[0], atol=1e-5, rtol=0)


def test_cross_attention_alone_equals_padded() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, N_HEADS).eval()
    memory, real = _batch(2)
    tgt = torch.randn(len(LENGTHS), 6, D_MODEL)
    n = LENGTHS[TARGET]

    alone = mha(tgt[TARGET : TARGET + 1], memory[TARGET : TARGET + 1, :n], memory[TARGET : TARGET + 1, :n])[0]
    batched = mha(tgt, memory, memory, attention_mask(real))[0]

    torch.testing.assert_close(batched[TARGET], alone[0], atol=1e-5, rtol=0)


def test_invariance_check_fails_without_pad_mask() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, N_HEADS).eval()
    x, _ = _batch(1)
    n = LENGTHS[TARGET]
    alone = mha(x[:1, :n], x[:1, :n], x[:1, :n])[0]
    batched = mha(x, x, x)[0]
    assert (batched[TARGET, :n] - alone[0]).abs().max() > 1e-2


@pytest.mark.parametrize("pre_norm", [True, False])
def test_full_model_alone_equals_padded(pre_norm: bool) -> None:
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(src_vocab=40, tgt_vocab=40, d_model=D_MODEL, n_heads=N_HEADS,
                                          d_ff=64, n_enc=2, n_dec=2, dropout=0.0, pre_norm=pre_norm)).eval()
    g = torch.Generator().manual_seed(3)
    src_lens, tgt_lens = [4, 9, 2], [6, 3, 8]
    src = torch.full((3, max(src_lens)), PAD)
    tgt = torch.full((3, max(tgt_lens)), PAD)
    for i, (s, t) in enumerate(zip(src_lens, tgt_lens)):
        src[i, :s] = torch.randint(4, 40, (s,), generator=g)
        tgt[i, :t] = torch.randint(4, 40, (t,), generator=g)

    batched = model(src, tgt)
    for i, (s, t) in enumerate(zip(src_lens, tgt_lens)):
        alone = model(src[i : i + 1, :s], tgt[i : i + 1, :t])
        torch.testing.assert_close(batched[i, :t], alone[0], atol=1e-5, rtol=0)
