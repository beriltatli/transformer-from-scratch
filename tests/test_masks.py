import pytest
import torch

from model.attention import MultiHeadAttention, scaled_dot_product_attention
from model.masks import attention_mask, causal_mask, pad_mask


def test_causal_mask_values() -> None:
    expected = torch.tensor(
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]], dtype=torch.bool
    )
    assert torch.equal(causal_mask(4), expected)


def test_pad_mask_marks_real_tokens() -> None:
    tokens = torch.tensor([[5, 6, 0], [7, 0, 0]])
    assert torch.equal(pad_mask(tokens, pad_id=0), torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool))


def test_shapes() -> None:
    pad = torch.ones(2, 5, dtype=torch.bool)
    assert attention_mask(pad).shape == (2, 1, 1, 5)
    assert attention_mask(None, causal_mask(5)).shape == (1, 1, 5, 5)
    assert attention_mask(pad, causal_mask(5)).shape == (2, 1, 5, 5)
    assert attention_mask() is None


def test_combination_is_elementwise_and() -> None:
    pad = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    causal = causal_mask(4)
    combined = attention_mask(pad, causal)
    for b in range(2):
        for t in range(4):
            for s in range(4):
                assert combined[b, 0, t, s] == (pad[b, s] and causal[t, s]), (b, t, s)


def test_combination_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        attention_mask(torch.ones(2, 5, dtype=torch.bool), causal_mask(4))


def test_attention_rejects_2d_mask_even_when_it_would_broadcast() -> None:
    # B == T == 3: a (B, S) mask would broadcast against (B, H, T, S) without error and mask
    # query rows by batch index. It must be refused instead.
    q = k = v = torch.randn(3, 2, 3, 4)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, torch.ones(3, 3, dtype=torch.bool))


def test_attention_rejects_float_mask() -> None:
    q = k = v = torch.randn(1, 1, 3, 4)
    with pytest.raises(TypeError):
        scaled_dot_product_attention(q, k, v, torch.zeros(1, 1, 3, 3))


def test_all_masked_row_gives_zero_weights_and_finite_gradients() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 2, 3, 4, requires_grad=True)
    k = torch.randn(2, 2, 3, 4, requires_grad=True)
    v = torch.randn(2, 2, 3, 4, requires_grad=True)
    # Sequence 1 is entirely padding: every one of its query rows sees no key.
    pad = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    out, weights = scaled_dot_product_attention(q, k, v, attention_mask(pad))

    assert torch.isfinite(out).all()
    assert torch.equal(weights[1], torch.zeros_like(weights[1]))
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    torch.testing.assert_close(weights[0].sum(-1), torch.ones(2, 3))

    out.sum().backward()
    for grad in (q.grad, k.grad, v.grad):
        assert torch.isfinite(grad).all()
    # The fully padded sequence had no path into the loss at all.
    assert torch.equal(q.grad[1], torch.zeros_like(q.grad[1]))


def test_all_masked_row_through_projection_is_output_bias() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(8, 2).eval()
    x = torch.randn(1, 3, 8)
    out, _ = mha(x, x, x, attention_mask(torch.zeros(1, 3, dtype=torch.bool)))
    torch.testing.assert_close(out[0], mha.out_proj.bias.expand(3, 8))
