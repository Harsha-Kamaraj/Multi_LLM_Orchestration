"""Adversarial fixtures must be detectable — that is their entire purpose.

Each case here corresponds to a corruption that produces plausible numbers
rather than an exception. The assertion is always the same shape: *something*
in the contract layer catches it. A case that nothing catches is a hole in the
contract, and this file is where that hole becomes visible.
"""

from __future__ import annotations

import pytest

from schemas import SchemaVersionError, assert_single_version, iter_errors
from schemas.adversarial import CASES, all_cases


def test_every_case_builds():
    cases = all_cases()
    assert set(cases) == set(CASES)
    for name, (rows, why) in cases.items():
        assert rows, f"{name} produced no rows"
        assert why, f"{name} has no explanation"


def test_mixed_versions_are_detected():
    rows, why = CASES["mixed_schema_versions"]()
    with pytest.raises(SchemaVersionError, match="mixes schema versions"):
        assert_single_version(rows)


def test_clean_store_passes_the_version_check(rows):
    assert assert_single_version(rows) == 1


def test_duplicate_ids_are_detectable(rows):
    dupes, why = CASES["duplicate_rollout_ids"]()
    ids = [r["rollout_id"] for r in dupes]
    assert len(ids) != len(set(ids)), why


def test_split_leakage_is_detectable():
    rows, why = CASES["split_leakage"]()
    by_task: dict[str, set[str]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], set()).add(row["split"])
    straddling = [t for t, s in by_task.items() if len(s) > 1]
    assert straddling, why


def test_half_graded_pair_fails_validation():
    rows, why = CASES["half_graded_pair"]()
    assert list(iter_errors(rows)), why


def test_orphan_ladder_step_fails_validation():
    rows, why = CASES["orphan_ladder_step"]()
    errors = [str(e) for e in iter_errors(rows)]
    assert any("parent_rollout_id" in e for e in errors), why


def test_batched_serving_row_fails_validation():
    rows, why = CASES["serving_row_batched"]()
    errors = [str(e) for e in iter_errors(rows)]
    assert any("queue depth" in e for e in errors), why


def test_impossible_counts_fail_validation():
    rows, why = CASES["negative_and_impossible_counts"]()
    errors = [str(e) for e in iter_errors(rows)]
    assert any("exceeds" in e for e in errors), why
    assert any("below minimum" in e for e in errors), why


def test_single_arm_store_is_detectable():
    """Structurally valid, and still unusable for a paired comparison. Caught by
    a completeness check rather than by validation — which is why the check has
    to exist somewhere other than the validator."""
    rows, why = CASES["single_arm_only"]()
    assert not list(iter_errors(rows)), "single-arm rows are individually valid"
    assert {r["arm"] for r in rows} == {"small"}, why


def test_ungraded_rows_are_valid_but_incomplete():
    """The dangerous case: nothing is malformed. A consumer must notice the
    nulls itself rather than relying on validation to raise."""
    rows, why = CASES["ungraded_rows_as_zero"]()
    assert not list(iter_errors(rows))
    ungraded = [r for r in rows if r["hidden_total"] is None]
    assert ungraded, why


def test_unicode_survives_a_round_trip():
    import json

    rows, why = CASES["unicode_and_nulls"]()
    assert json.loads(json.dumps(rows)) == rows, why


def test_dirty_run_ids_are_flagged_not_rejected():
    """Readable during development, non-publishable in a report. The distinction
    has to live in the publishing gate, not the validator."""
    rows, why = CASES["dirty_run_id"]()
    assert not list(iter_errors(rows))
    assert all(r["run_id"].endswith("-dirty") for r in rows), why
