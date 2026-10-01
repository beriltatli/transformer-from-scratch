"""Exact-zero proof that position t cannot see positions > t, nor any other batch element.

It runs twice: on a bare stack of causal self-attention blocks, where a failure points at the
attention or the mask, and on the assembled Transformer from raw target embeddings to logits,
which covers scaling, positions, cross-attention, feed-forward, norms and the tied generator.

The detector is only worth something if it fires on broken models, so the second half of this
file builds four realistic leaks and asserts each one is caught.
"""

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.attention import MultiHeadAttention
from model.masks import attention_mask, causal_mask, pad_mask
from model.transformer import Transformer, TransformerConfig

Forward = Callable[[Tensor], Tensor]
BATCH, LENGTH, D_MODEL, PAD_ID = 3, 7, 16, 0


class CausalStack(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, diagonal: int = 0, causal: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(MultiHeadAttention(D_MODEL, n_heads) for _ in range(n_layers))
        self.diagonal = diagonal
        self.causal = causal

    def forward(self, x: Tensor, pad: Tensor | None = None) -> Tensor:
        causal = causal_mask(x.size(1))
        if self.diagonal:
            causal = torch.ones_like(causal).tril(self.diagonal)
        mask = attention_mask(pad, causal if self.causal else None)
        for layer in self.layers:
            x = x + layer(x, x, x, mask)[0]
        return x


def leak_map(forward: Forward, x: Tensor) -> Tensor:
    """leak[b, t] = largest |d out[b, t, :] / d x[b', s, :]| over every (b', s) that b, t must not see.

    One backward pass per (b, t), so each output is isolated. Summing out[b, t, :] is enough:
    if any coordinate depended on a forbidden input, the sum would too, barring an exact
    cancellation that random weights make vanishingly unlikely.
    """
    x = x.detach().requires_grad_(True)
    out = forward(x)
    b_n, t_n = out.shape[:2]
    leak = torch.zeros(b_n, t_n, dtype=torch.float64)
    for b in range(b_n):
        for t in range(t_n):
            (grad,) = torch.autograd.grad(out[b, t].sum(), x, retain_graph=True)
            forbidden = torch.ones(b_n, t_n, dtype=torch.bool)
            forbidden[b, : t + 1] = False
            leak[b, t] = grad[forbidden].abs().max() if forbidden.any() else 0.0
    return leak


def _inputs(seed: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, LENGTH, D_MODEL, generator=g)
    tokens = torch.randint(1, 50, (BATCH, LENGTH), generator=g)
    tokens[1, 5:] = PAD_ID
    tokens[2, 2:] = PAD_ID
    return x, tokens


@pytest.mark.parametrize("n_heads", [1, 4, 8])
@pytest.mark.parametrize("n_layers", [1, 3])
@pytest.mark.parametrize("padded", [False, True])
def test_future_gradient_is_exactly_zero(n_heads: int, n_layers: int, padded: bool) -> None:
    torch.manual_seed(0)
    model = CausalStack(n_layers, n_heads).eval()
    x, tokens = _inputs(1)
    pad = pad_mask(tokens, PAD_ID) if padded else None
    leak = leak_map(lambda inp: model(inp, pad), x)
    # == 0, not allclose: a masked key's weight is exp(-inf) = 0.0, so the true derivative is
    # exactly zero and any non-zero value means a path exists.
    assert (leak == 0).all(), f"leak at (b, t) = {(leak != 0).nonzero().tolist()}"


@pytest.mark.parametrize("n_layers", [1, 3])
def test_perturbing_future_leaves_past_bit_identical(n_layers: int) -> None:
    torch.manual_seed(0)
    model = CausalStack(n_layers, n_heads=4).eval()
    x, _ = _inputs(2)
    base = model(x)
    g = torch.Generator().manual_seed(3)
    for t in range(LENGTH - 1):
        perturbed = x.clone()
        perturbed[:, t + 1 :] = 1e3 * torch.randn(BATCH, LENGTH - t - 1, D_MODEL, generator=g)
        out = model(perturbed)
        assert torch.equal(out[:, : t + 1], base[:, : t + 1]), f"past changed when perturbing > {t}"
        assert not torch.equal(out[:, t + 1 :], base[:, t + 1 :]), "perturbation had no effect at all"


def _max_leak(model: nn.Module) -> float:
    x, _ = _inputs(4)
    return float(leak_map(model.eval(), x).max())


def test_detector_catches_missing_mask() -> None:
    torch.manual_seed(0)
    assert _max_leak(CausalStack(2, 4, causal=False)) > 1e-3


def test_detector_catches_off_by_one_diagonal() -> None:
    torch.manual_seed(0)
    assert _max_leak(CausalStack(1, 4, diagonal=1)) > 1e-3


def test_detector_catches_head_split_without_transpose(monkeypatch: pytest.MonkeyPatch) -> None:
    # Right shape, wrong memory order: rows of (Dh) values from different time steps end up
    # sharing a "position". No mask can stop this, which is why the test runs on the model
    # and not on the mask.
    def bad_split(self: MultiHeadAttention, x: Tensor) -> Tensor:
        b, length, _ = x.shape
        return x.reshape(b, self.n_heads, length, self.d_head)

    monkeypatch.setattr(MultiHeadAttention, "_split", bad_split)
    torch.manual_seed(0)
    assert _max_leak(CausalStack(1, 4)) > 1e-3


def test_detector_catches_soft_mask_that_allclose_would_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    # An additive -20 instead of -inf. exp(-20) ~ 2e-9: the leak is real, and an
    # assert_close(atol=1e-6) leakage test would call it zero.
    def soft_sdpa(q, k, v, mask=None, dropout_p=0.0, training=False):
        scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
        scores = scores + (~mask) * -20.0
        weights = F.softmax(scores, dim=-1)
        return weights @ v, weights

    monkeypatch.setattr("model.attention.scaled_dot_product_attention", soft_sdpa)
    torch.manual_seed(0)
    leak = _max_leak(CausalStack(1, 4))
    assert 0 < leak < 1e-6, leak


def _transformer(pre_norm: bool, positions: str) -> Transformer:
    torch.manual_seed(0)
    cfg = TransformerConfig(src_vocab=50, tgt_vocab=60, d_model=D_MODEL, n_heads=4, d_ff=32,
                            n_enc=2, n_dec=3, dropout=0.0, pre_norm=pre_norm, positions=positions, max_len=32)
    return Transformer(cfg).eval()


def _source() -> Tensor:
    src = torch.randint(4, 50, (BATCH, 6), generator=torch.Generator().manual_seed(5))
    src[1, 4:] = PAD_ID
    return src


@pytest.mark.parametrize("pre_norm", [True, False])
@pytest.mark.parametrize("positions", ["sinusoidal", "learned"])
@pytest.mark.parametrize("padded", [False, True])
def test_transformer_future_gradient_is_exactly_zero(pre_norm: bool, positions: str, padded: bool) -> None:
    model = _transformer(pre_norm, positions)
    memory, src_pad = model.encode(_source())
    memory = memory.detach()
    x, tokens = _inputs(6)
    tgt_pad = pad_mask(tokens, PAD_ID) if padded else None
    # Memory is a constant here, so cross-batch leakage through it is not what is being tested;
    # it is shared per row and cannot carry another row's target.
    leak = leak_map(lambda inp: model.decode_embedded(inp, memory, src_pad, tgt_pad)[0], x)
    assert (leak == 0).all(), f"leak at (b, t) = {(leak != 0).nonzero().tolist()}"


def test_transformer_future_tokens_leave_past_logits_bit_identical() -> None:
    model = _transformer(pre_norm=True, positions="sinusoidal")
    src = _source()
    g = torch.Generator().manual_seed(7)
    tgt = torch.randint(4, 60, (BATCH, LENGTH), generator=g)
    base = model(src, tgt)
    for t in range(LENGTH - 1):
        changed = tgt.clone()
        changed[:, t + 1 :] = torch.randint(4, 60, (BATCH, LENGTH - t - 1), generator=g)
        out = model(src, changed)
        assert torch.equal(out[:, : t + 1], base[:, : t + 1]), f"logits <= {t} changed"


def test_transformer_output_depends_on_source() -> None:
    # The other half of the contract: the decoder must see the encoder. A model that passed
    # every leakage test by ignoring its inputs would be useless.
    model = _transformer(pre_norm=True, positions="sinusoidal")
    tgt = torch.randint(4, 60, (BATCH, LENGTH), generator=torch.Generator().manual_seed(8))
    src = _source()
    other = src.clone()
    other[:, 0] = (other[:, 0] + 1 - 4) % 46 + 4
    assert (model(src, tgt) - model(other, tgt)).abs().max() > 1e-3
