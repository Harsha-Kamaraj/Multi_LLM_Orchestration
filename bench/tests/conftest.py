"""Shared fixtures for R1's tests.

Everything here runs without a GPU, a network, or an API key. That is the
point: the sweep runner, the store, resume, sealing, and the cost model are
exercised end to end on every commit, on a laptop, in seconds. A pipeline whose
only integration test needs a GPU is a pipeline that gets tested when someone
remembers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Importable without `pip install -e .`, so a fresh clone can run the suite
# before it has installed anything.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(autouse=True)
def _no_tokenizer_downloads(monkeypatch):
    """Keep every test off the Hugging Face hub.

    Without this, a test that touches `get_tokenizer` on an unknown model id
    imports torch and may reach for the network — turning a millisecond unit
    test into a ten-second one, or a hang on an offline machine.
    """
    monkeypatch.setenv("ORCH_TOKENIZER", "approx")
    from orchestrator.workers import tokenization

    tokenization.clear_cache()
    yield
    tokenization.clear_cache()


def write_corpus(path: Path, n: int = 24, *, with_solutions: bool = True,
                 splits: tuple[str, ...] = ("train", "val", "test")) -> Path:
    """Write a synthetic task manifest in R2's expected shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            record = {
                "task_id": f"mbpp/{i}",
                "prompt": f"Write a function add_{i}(a, b) that returns a + b.",
                "entrypoint": f"add_{i}",
                "visible_tests": f"def test_add():\n    assert add_{i}(1, 2) == 3",
                "tests": (
                    f"def test_add():\n    assert add_{i}(1, 2) == 3\n"
                    f"def test_neg():\n    assert add_{i}(-1, 1) == 0"
                ),
                "split": splits[i % len(splits)],
                "dataset": "mbpp+",
            }
            if with_solutions:
                record["mock_solution"] = f"def add_{i}(a, b):\n    return a + b"
            fh.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "data" / "tasks" / "pilot.jsonl")


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def sweep_config(corpus_path: Path, runs_root: Path):
    """A small, fast, fully deterministic sweep."""
    from orchestrator.workers.sweep import SweepConfig

    return SweepConfig(
        tasks_path=corpus_path,
        out_root=runs_root,
        arms=("direct_small", "direct_large"),
        seeds=(0, 1),
        backend="mock",
        batch_size=8,
        # No imputation during unit tests: cost coefficients are a separate
        # concern and their absence must not change what a sweep writes.
        coefficients_path=None,
        dataset="mbpp+",
    )
