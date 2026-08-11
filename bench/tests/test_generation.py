"""The `Generation` record — identity, immutability, and the anti-conflation
fields that stop sweep wall-clock being read as latency."""

from __future__ import annotations

import pytest

from orchestrator.workers.generation import (
    FINISH_REASONS, SCHEMA_VERSION, Generation, normalize_finish_reason,
)

RUN_ID = "2026-08-11-abc1234-def456"


def gen(**kwargs):
    base = dict(run_id=RUN_ID, task_id="t0", arm="direct_small", seed=0,
                params_hash="c0ff9e5c88e4")
    return Generation(**{**base, **kwargs})


def test_rollout_id_is_deterministic():
    """Derived, not random, so a re-run under identical conditions produces
    the same id and a child can reference a parent not yet written."""
    assert gen().rollout_id == gen().rollout_id


@pytest.mark.parametrize("field,value", [
    ("run_id", "2026-08-12-abc1234-def456"),
    ("task_id", "t1"),
    ("arm", "direct_large"),
    ("seed", 1),
    ("params_hash", "aaaaaaaaaaaa"),
    ("ladder_step", 1),
])
def test_rollout_id_covers_every_identity_field(field, value):
    assert gen(**{field: value}).rollout_id != gen().rollout_id


def test_cell_key_is_the_resume_key():
    assert gen().cell_key == ("t0", "direct_small", 0, "c0ff9e5c88e4")


def test_truncation_is_flagged():
    """`finish_reason == "length"` grades as a failure and looks exactly like
    a model capability gap."""
    assert gen(finish_reason="length").truncated
    assert not gen(finish_reason="stop").truncated


def test_failed_covers_errors_refusals_and_empty_code():
    assert gen(finish_reason="error").failed
    assert gen(finish_reason="refusal").failed
    assert gen(finish_reason="stop", code="").failed
    assert not gen(finish_reason="stop", code="x = 1").failed


def test_with_cost_does_not_mutate():
    """A row is never mutated in place; the store enforces the same at the
    file level."""
    original = gen()
    updated = original.with_cost(1.5, 2.5)
    assert original.gpu_seconds is None
    assert updated.gpu_seconds == 1.5 and updated.imputed_latency_s == 2.5
    assert updated.rollout_id == original.rollout_id


def test_row_carries_mode_and_batch_size():
    """Together these are what make a latency filter possible downstream."""
    row = gen(mode="sweep", batch_size=256).to_row()
    assert row["mode"] == "sweep" and row["batch_size"] == 256


def test_row_carries_null_grading_fields():
    row = gen().to_row()
    for field in ("visible_passed", "visible_total", "hidden_passed",
                  "hidden_total", "error_class", "hack_flags", "grade_duration_s"):
        assert row[field] is None


def test_row_round_trips_ignoring_other_roles_fields():
    row = gen(code="x = 1").to_row()
    row.update({"visible_passed": 3, "hack_flags": ["hardcoded"]})
    restored = Generation.from_row(row)
    assert restored.code == "x = 1"
    assert restored.rollout_id == gen(code="x = 1").rollout_id


def test_every_row_carries_a_schema_version():
    """A mixed-version store must be detectable, never silently averaged."""
    assert gen().to_row()["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("raw,expected", [
    ("stop", "stop"), ("end_turn", "stop"), ("stop_sequence", "stop"),
    ("length", "length"), ("max_tokens", "length"),
    ("refusal", "refusal"), ("content_filter", "refusal"),
    ("abort", "error"), ("error", "error"),
    ("something_new", "unknown"), (None, "unknown"), ("STOP", "stop"),
])
def test_finish_reasons_normalize(raw, expected):
    assert normalize_finish_reason(raw) == expected


def test_normalization_always_lands_in_the_vocabulary():
    """A new value from a backend upgrade shows up as a visible bucket, not
    as a new string nobody is counting."""
    for raw in ("weird", "", None, "TOOL_USE", "  stop  "):
        assert normalize_finish_reason(raw) in FINISH_REASONS


def test_error_normalizes_to_error_not_unknown():
    """Regression: without this, a recorded failure normalizes to "unknown",
    `failed` stops recognising it, and a sweep reports zero failures while
    writing them."""
    assert normalize_finish_reason("error") == "error"
    assert gen(finish_reason=normalize_finish_reason("error")).failed
