"""The validator must reject what JSON Schema alone would wave through."""

from __future__ import annotations

import copy

import pytest

from schemas import ValidationError, validate_rollout, validate_task


def test_synthetic_store_is_schema_valid(rows):
    """The fixture generator must not be able to emit an invalid row.

    If this fails, every downstream test is running against data the real
    pipeline would reject, and their passing means nothing.
    """
    errors = [str(e) for e in __import__("schemas").iter_errors(rows)]
    assert errors == []


def test_null_is_not_a_permitted_integer(rows):
    row = copy.deepcopy(rows[0])
    row["seed"] = None
    with pytest.raises(ValidationError, match="expected type integer"):
        validate_rollout(row)


def test_bool_is_not_an_integer(rows):
    """Python says bool is an int. JSON Schema does not, and a True seed that
    validates becomes seed 1 in every downstream group-by."""
    row = copy.deepcopy(rows[0])
    row["seed"] = True
    with pytest.raises(ValidationError, match="expected type integer"):
        validate_rollout(row)


def test_passed_without_total_is_rejected(rows):
    row = copy.deepcopy(rows[0])
    row["hidden_total"] = None
    with pytest.raises(ValidationError, match="graded together"):
        validate_rollout(row)


def test_passed_exceeding_total_is_rejected(rows):
    row = copy.deepcopy(rows[0])
    row["hidden_passed"] = row["hidden_total"] + 1
    with pytest.raises(ValidationError, match="exceeds"):
        validate_rollout(row)


def test_ungraded_row_is_valid(rows):
    """Null grading fields are legitimate before R2 runs — the store must be
    readable between the sweep and the grade."""
    row = copy.deepcopy(rows[0])
    for key in ("visible_passed", "visible_total", "hidden_passed", "hidden_total"):
        row[key] = None
    validate_rollout(row)


def test_orphan_ladder_step_is_rejected(rows):
    row = copy.deepcopy(rows[0])
    row["ladder_step"] = 2
    row["parent_rollout_id"] = None
    with pytest.raises(ValidationError, match="parent_rollout_id"):
        validate_rollout(row)


def test_step_zero_with_parent_is_rejected(rows):
    row = copy.deepcopy(rows[0])
    row["ladder_step"] = 0
    row["parent_rollout_id"] = "deadbeefdeadbeef"
    with pytest.raises(ValidationError, match="must not have a parent"):
        validate_rollout(row)


def test_batched_serving_row_is_rejected(rows):
    """The category error the mode/batch_size pair exists to catch."""
    row = copy.deepcopy(rows[0])
    row["mode"] = "serving"
    row["batch_size"] = 16
    with pytest.raises(ValidationError, match="queue depth"):
        validate_rollout(row)


def test_run_id_pattern_is_enforced(rows):
    row = copy.deepcopy(rows[0])
    row["run_id"] = "yesterday-ish"
    with pytest.raises(ValidationError, match="does not match"):
        validate_rollout(row)


def test_dirty_run_id_is_structurally_valid(rows):
    """-dirty parses; refusing to *publish* it is a separate, explicit check.
    Conflating the two would make dirty runs unreadable during development."""
    row = copy.deepcopy(rows[0])
    row["run_id"] = row["run_id"] + "-dirty"
    validate_rollout(row)


def test_unknown_split_is_rejected(rows):
    row = copy.deepcopy(rows[0])
    row["split"] = "holdout"
    with pytest.raises(ValidationError, match="not in"):
        validate_rollout(row)


def test_unknown_finish_reason_is_rejected(rows):
    row = copy.deepcopy(rows[0])
    row["finish_reason"] = "content_filter"
    with pytest.raises(ValidationError, match="not in"):
        validate_rollout(row)


def test_difficulty_requires_provenance():
    task = {
        "task_id": "t/1", "dataset": "d", "prompt": "p", "entrypoint": "f",
        "visible_tests": "", "hidden_tests": "", "difficulty": 0.8,
    }
    with pytest.raises(ValidationError, match="leakage"):
        validate_task(task)

    task["difficulty_provenance"] = "structural"
    validate_task(task)


def test_iter_errors_reports_all_not_just_first(rows):
    from schemas import iter_errors

    bad = [copy.deepcopy(rows[0]) for _ in range(3)]
    for row in bad:
        row["split"] = "nope"
    assert len(list(iter_errors(bad))) == 3
