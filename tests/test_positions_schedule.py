import math

import pytest
import torch

from model.pos import LearnedPositions, SinusoidalPositions
from model.schedule import inverse_sqrt_factor


def test_sinusoidal_known_values() -> None:
    pe = SinusoidalPositions(d_model=4, max_len=10).pe
    # dims (0, 1) have wavelength 2π, dims (2, 3) have frequency 10000^(-2/4) = 0.01.
    torch.testing.assert_close(pe[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    torch.testing.assert_close(pe[3], torch.tensor([math.sin(3), math.cos(3), math.sin(0.03), math.cos(0.03)]))


def test_sinusoidal_dot_product_depends_only_on_offset() -> None:
    pe = SinusoidalPositions(d_model=64, max_len=200).pe
    for k in (1, 5, 17):
        dots = torch.stack([pe[p] @ pe[p + k] for p in range(0, 150, 7)])
        torch.testing.assert_close(dots, dots[0].expand_as(dots), atol=1e-4, rtol=0)


def test_learned_positions_refuse_lengths_beyond_table() -> None:
    pos = LearnedPositions(d_model=8, max_len=5)
    pos(torch.zeros(1, 5, 8))
    with pytest.raises(ValueError):
        pos(torch.zeros(1, 6, 8))


def test_schedule_shape() -> None:
    assert inverse_sqrt_factor(1, warmup=100) == pytest.approx(0.01)
    assert inverse_sqrt_factor(50, warmup=100) == pytest.approx(0.5)
    assert inverse_sqrt_factor(100, warmup=100) == pytest.approx(1.0)
    assert inverse_sqrt_factor(400, warmup=100) == pytest.approx(0.5)


def test_schedule_without_ramp_differs_only_before_warmup() -> None:
    assert inverse_sqrt_factor(1, warmup=100, ramp=False) == 1.0
    for step in (100, 101, 1000, 10_000):
        assert inverse_sqrt_factor(step, 100, ramp=False) == inverse_sqrt_factor(step, 100, ramp=True)


def test_schedule_matches_vaswani_formula() -> None:
    d, warmup = 512, 4000
    peak = (d * warmup) ** -0.5
    for step in (1, 1000, 4000, 20000):
        vaswani = d**-0.5 * min(step**-0.5, step * warmup**-1.5)
        assert peak * inverse_sqrt_factor(step, warmup) == pytest.approx(vaswani)
