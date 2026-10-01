import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def test_python_matches_pin() -> None:
    pinned = (ROOT / ".python-version").read_text().strip()
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == pinned


def test_torch_matches_pin() -> None:
    pins = dict(
        line.split("==") for line in (ROOT / "requirements.txt").read_text().split() if "==" in line
    )
    # CI installs the +cpu local build; the public version must still match.
    assert torch.__version__.split("+")[0] == pins["torch"]
