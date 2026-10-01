"""The only file allowed to import nn.MultiheadAttention: it is the numerical reference."""

import pytest
import torch
from torch import nn

from model.attention import MultiHeadAttention
from model.masks import attention_mask, causal_mask

N_CASES = 200
MASK_KINDS = ("none", "pad", "causal", "both")


def _copy_weights(mine: MultiHeadAttention, ref: nn.MultiheadAttention) -> None:
    # The reference stores q, k, v as one (3D, D) matrix stacked in that order.
    d = mine.d_model
    with torch.no_grad():
        for i, proj in enumerate((mine.q_proj, mine.k_proj, mine.v_proj)):
            proj.weight.copy_(ref.in_proj_weight[i * d : (i + 1) * d])
            proj.bias.copy_(ref.in_proj_bias[i * d : (i + 1) * d])
        mine.out_proj.weight.copy_(ref.out_proj.weight)
        mine.out_proj.bias.copy_(ref.out_proj.bias)


def _case(i: int) -> dict:
    g = torch.Generator().manual_seed(i)

    def pick(lo: int, hi: int) -> int:
        return int(torch.randint(lo, hi + 1, (1,), generator=g))

    n_heads = (i % 8) + 1
    kind = MASK_KINDS[(i // 8) % 4]
    batch = pick(1, 4)
    tgt_len = pick(1, 9)
    # Causal masking implies self-attention, so S == T; otherwise exercise cross-attention too.
    src_len = tgt_len if kind in ("causal", "both") else pick(1, 9)
    d_model = n_heads * pick(1, 6)

    pad = None
    if kind in ("pad", "both"):
        # Right padding with length >= 1: every query keeps at least key 0, so no row is fully
        # masked. The reference returns nan for those rows; that case lives in test_masks.py.
        lengths = torch.randint(1, src_len + 1, (batch,), generator=g)
        pad = torch.arange(src_len)[None, :] < lengths[:, None]
    causal = causal_mask(tgt_len) if kind in ("causal", "both") else None

    return dict(
        seed=i, n_heads=n_heads, kind=kind, d_model=d_model, pad=pad, causal=causal,
        query=torch.randn(batch, tgt_len, d_model, generator=g),
        key=torch.randn(batch, src_len, d_model, generator=g),
        value=torch.randn(batch, src_len, d_model, generator=g),
    )


@pytest.mark.parametrize("i", range(N_CASES))
def test_matches_reference(i: int) -> None:
    c = _case(i)
    torch.manual_seed(c["seed"])
    ref = nn.MultiheadAttention(c["d_model"], c["n_heads"], batch_first=True).eval()
    mine = MultiHeadAttention(c["d_model"], c["n_heads"]).eval()
    _copy_weights(mine, ref)

    # The reference's boolean masks mean True = blocked; ours mean True = may attend.
    ref_out, ref_weights = ref(
        c["query"], c["key"], c["value"],
        key_padding_mask=None if c["pad"] is None else ~c["pad"],
        attn_mask=None if c["causal"] is None else ~c["causal"],
        need_weights=True, average_attn_weights=False,
    )
    out, weights = mine(c["query"], c["key"], c["value"], attention_mask(c["pad"], c["causal"]))

    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=0)
    torch.testing.assert_close(weights, ref_weights, atol=1e-5, rtol=0)


def test_cases_cover_every_head_count_and_mask_kind() -> None:
    seen = {(_case(i)["n_heads"], _case(i)["kind"]) for i in range(N_CASES)}
    assert seen == {(h, k) for h in range(1, 9) for k in MASK_KINDS}
