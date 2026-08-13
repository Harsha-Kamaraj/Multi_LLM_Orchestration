"""Fixtures for the contract tests.

These run with no GPU, no network, and no serving stack — the contract must be
importable by a consumer who has installed none of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for path in (str(_ROOT), str(_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from schemas import SynthConfig, generate  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    """A small deterministic fixture store, built once."""
    return generate(SynthConfig(n_tasks=200, seeds=3), seed=11)


@pytest.fixture(scope="session")
def rows(synth):
    return synth.rows
