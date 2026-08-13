"""Fixtures for R4's tests.

No GPU, no network, no rollout store on disk. Everything runs against the
synthetic generator, whose answer is known by construction — which is the only
way to validate a statistics harness. Code that has never been run against a
known answer is untested code that produces numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for path in (str(_ROOT), str(_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval import load_rows  # noqa: E402
from schemas import SynthConfig, generate  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    """A fixture store large enough for stable intervals, small enough to be fast."""
    return generate(SynthConfig(n_tasks=300, seeds=3), seed=23)


@pytest.fixture(scope="session")
def store(synth):
    """Loaded across train+val. `test` stays locked, as it does in real use."""
    return load_rows(synth.rows)


@pytest.fixture(scope="session")
def all_splits(synth):
    """Every row, bypassing the split filter — for tests about loading itself."""
    return load_rows(synth.rows, splits=("train", "val", "test"),
                     unlock=_fake_unlock())


def _fake_unlock():
    from eval.loading import TestSplitUnlock

    return TestSplitUnlock(reason="test suite", preregistration=__file__)
