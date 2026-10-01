import pytest

from scripts.config import load_config
from scripts.overfit import run_overfit


@pytest.mark.slow
def test_ten_pairs_memorised_and_regenerated_exactly() -> None:
    result = run_overfit(load_config())
    # Mean token NLL over the last 10 steps: one step is a single noisy sample of a batch that
    # is the whole dataset, but at this point it should not be noisy at all.
    final = sum(r["nll"] for r in result.history[-10:]) / 10
    assert final < 0.01, final
    assert result.hypotheses == result.references
