"""Shared fixtures for R3's tests.

`raw_row` builds its row by calling R1's `Generation.to_row()` rather than by
hand-writing a dict. That is deliberate: it makes these tests a drift canary.
If R1 renames a column or changes a type, R3's contract tests fail on the next
commit instead of R3 discovering it against a store in week four.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from orchestrator.workers.generation import Generation


@pytest.fixture
def raw_row() -> Callable[..., dict[str, Any]]:
    """A row exactly as R1 writes it, with optional overrides.

    Grading columns come back as nulls, which is what an ungraded sweep
    actually looks like today — no grader writes them yet.
    """

    def build(**overrides: Any) -> dict[str, Any]:
        gen = Generation(
            run_id="2026-08-12-abc1234-def567",
            task_id="mbpp/1",
            arm="direct_small",
            seed=0,
            params_hash="p" * 12,
            text="```python\ndef f():\n    return 1\n```",
            code="def f():\n    return 1\n",
            model_id="mock-small",
            prefill_tokens=120,
            decode_tokens=48,
            wall_ms=310.0,
            finish_reason="stop",
            backend="mock",
            extract_strategy="fenced",
            code_parses=True,
            split="train",
            dataset="mbpp+",
            code_version="abc1234",
        )
        row = gen.to_row()
        row.update(overrides)
        return row

    return build


@pytest.fixture
def graded_row(raw_row: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """A row R2 has already graded — what a training example looks like."""

    def build(**overrides: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "visible_passed": 3,
            "visible_total": 3,
            "hidden_passed": 8,
            "hidden_total": 8,
            "error_class": None,
            "hack_flags": [],
            "grade_duration_s": 0.42,
        }
        defaults.update(overrides)
        return raw_row(**defaults)

    return build
