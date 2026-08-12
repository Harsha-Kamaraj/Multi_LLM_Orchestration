from __future__ import annotations

import pytest

from orchestrator.graders.base import Grade, TestResult


def test_test_result_all_passed():
    assert TestResult(3, 3).all_passed
    assert not TestResult(2, 3).all_passed
    assert not TestResult(0, 0).all_passed  # zero tests is not "all passed"


def test_test_result_rate():
    assert TestResult(1, 4).rate == 0.25
    assert TestResult(0, 0).rate == 0.0


def test_grade_flat_accessors_match_generation_row_keys():
    g = Grade(visible=TestResult(1, 2), hidden=TestResult(3, 5))
    row = g.to_row()
    assert row == {
        "visible_passed": 1,
        "visible_total": 2,
        "hidden_passed": 3,
        "hidden_total": 5,
        "error_class": "none",
        "hack_flags": [],
        "grade_duration_s": 0.0,
    }


def test_solved_is_hidden_only():
    """`solved` must never be satisfiable by the visible tier alone —
    it is the label, and leaking it via a visible-only pass would defeat
    the entire point of the split."""
    g = Grade(visible=TestResult(2, 2), hidden=TestResult(1, 2))
    assert not g.solved
    g2 = Grade(visible=TestResult(0, 2), hidden=TestResult(2, 2))
    assert g2.solved


def test_grade_has_no_blended_scalar():
    """The whole point of the split: no single field spans both tiers."""
    g = Grade(visible=TestResult(1, 1), hidden=TestResult(1, 1))
    for forbidden in ("passed", "total", "score"):
        assert not hasattr(g, forbidden), (
            f"Grade must not expose a blended '{forbidden}' — "
            f"visible and hidden must stay structurally separate"
        )


def test_grade_rejects_unknown_error_class():
    with pytest.raises(ValueError):
        Grade(visible=TestResult(0, 1), hidden=TestResult(0, 1),
              error_class="not_a_real_class")
