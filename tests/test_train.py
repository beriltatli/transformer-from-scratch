import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.train import LENGTH_MULTIPLE, collate, label_smoothed_nll, make_batches, shift
from tokenizer.bpe import BOS, EOS, PAD


def _corpus(n: int, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    src = [np.arange(4, 4 + rng.randint(1, 30), dtype=np.int16) for _ in range(n)]
    tgt = [np.array([BOS, *range(4, 4 + rng.randint(1, 30)), EOS], dtype=np.int16) for _ in range(n)]
    return src, tgt


def test_batches_cover_every_example_once_within_budget() -> None:
    src, tgt = _corpus(500, 0)
    batches = make_batches(src, tgt, max_tokens=256, rng=random.Random(1))
    assert sorted(i for b in batches for i in b) == list(range(500))
    for idx in batches:
        s, t = collate(src, tgt, idx)
        assert s.size(0) * max(s.size(1), t.size(1) - 1) <= 256 or len(idx) == 1


def test_one_shape_per_group_and_bucketed_lengths() -> None:
    src, tgt = _corpus(2000, 2)
    shapes = set()
    for idx in make_batches(src, tgt, max_tokens=512, rng=random.Random(3)):
        s, t = collate(src, tgt, idx)
        tgt_in, _ = shift(t)
        assert s.size(1) % LENGTH_MULTIPLE == 0 and tgt_in.size(1) % LENGTH_MULTIPLE == 0
        shapes.add((s.size(1), tgt_in.size(1)))
    # 30 tokens max per side -> at most 4 buckets per side.
    assert len(shapes) <= 16


def test_shift_aligns_input_and_target() -> None:
    tgt = torch.tensor([[BOS, 7, 8, EOS], [BOS, 9, EOS, PAD]])
    tgt_in, tgt_out = shift(tgt)
    assert tgt_in.tolist() == [[BOS, 7, 8], [BOS, 9, EOS]]
    assert tgt_out.tolist() == [[7, 8, EOS], [9, EOS, PAD]]


def test_unsmoothed_loss_matches_cross_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    target = torch.tensor([[4, 5, 6, PAD, PAD], [7, 8, 9, 10, EOS]])
    loss, nll, n = label_smoothed_nll(logits, target, 0.0)
    ref = F.cross_entropy(logits.reshape(-1, 11), target.reshape(-1), ignore_index=PAD, reduction="sum")
    assert n == 8
    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(nll, ref)


def test_smoothed_loss_on_uniform_logits_is_log_vocab() -> None:
    # Uniform prediction: NLL and the uniform term are both log V, so any smoothing gives log V.
    loss, _, n = label_smoothed_nll(torch.zeros(1, 3, 20), torch.tensor([[4, 5, 6]]), 0.1)
    assert (loss / n).item() == pytest.approx(np.log(20))
