"""The contract must describe what R1 actually writes.

This is the test that stops the schema becoming aspirational. R1's
`Generation.to_row()` is the only producer of real rollout rows; if the schema
and that projection drift apart, the schema is documentation rather than a
contract, and every role is building against a different shape.

Two sources of truth for a contract is the same as none.
"""

from __future__ import annotations

import pytest

from schemas import SCHEMA_VERSION, rollout_schema, validate_rollout

generation = pytest.importorskip(
    "orchestrator.workers.generation",
    reason="R1's package is not installed; conformance is checked where it is",
)


def test_schema_version_matches_r1():
    """A version that drifts is worse than no version — it makes a mixed store
    look single-version."""
    assert generation.SCHEMA_VERSION == SCHEMA_VERSION, (
        "schemas/version.py and orchestrator.workers.generation disagree on "
        "SCHEMA_VERSION. Bump both in one commit or neither."
    )


def test_r1_default_row_satisfies_the_contract():
    row = generation.Generation(
        run_id="2026-01-01-0f1ced0-abc123",
        task_id="t/1",
        arm="small",
        seed=0,
        params_hash="deadbeef",
        split="train",
        finish_reason="stop",
    ).to_row()
    validate_rollout(row)


def test_r1_emits_every_required_field():
    """Required fields are the ones no consumer can work around. R1 must emit
    all of them, including the grading keys it leaves null."""
    row = generation.Generation(
        run_id="2026-01-01-0f1ced0-abc123",
        task_id="t/1", arm="small", seed=0, params_hash="x", split="val",
    ).to_row()
    missing = set(rollout_schema()["required"]) - set(row)
    assert missing == set(), f"R1's projection omits required fields: {missing}"


def test_grading_keys_are_present_and_null_before_grading():
    """A store read before grading must have the same columns as one read
    after, so nothing downstream special-cases an ungraded run."""
    row = generation.Generation(
        run_id="2026-01-01-0f1ced0-abc123",
        task_id="t/1", arm="small", seed=0, params_hash="x", split="test",
    ).to_row()
    for key in ("visible_passed", "visible_total", "hidden_passed",
                "hidden_total", "error_class", "hack_flags", "grade_duration_s"):
        assert key in row, f"{key} missing before grading"
        assert row[key] is None, f"{key} should be null before grading"


def test_finish_reason_vocabulary_agrees():
    """R1 normalizes every backend's finish reason onto a shared vocabulary.
    The schema enumerates it. A value R1 can emit but the schema rejects would
    fail validation only once that backend was used in anger."""
    schema_values = set(rollout_schema()["properties"]["finish_reason"]["enum"])
    assert set(generation.FINISH_REASONS) == schema_values


def test_mode_vocabulary_agrees():
    schema_values = set(rollout_schema()["properties"]["mode"]["enum"])
    assert schema_values == {"sweep", "serving"}


def test_rollout_id_derivation_agrees():
    """Both sides derive the id from the same tuple. If they diverge, joins
    between a fixture and a real store silently produce empty results."""
    from schemas.synth import _rollout_id

    gen = generation.Generation(
        run_id="2026-01-01-0f1ced0-abc123",
        task_id="t/1", arm="small", seed=2, params_hash="ph", split="train",
        ladder_step=0,
    )
    row = gen.to_row()
    assert _rollout_id(row) == gen.rollout_id
