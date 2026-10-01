import pytest
import torch

from model.attention import MultiHeadAttention
from model.masks import attention_mask, causal_mask

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
