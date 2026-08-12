"""The row contract, and the guards that make leakage structural."""

from __future__ import annotations

import pytest

from orchestrator.policy import contract
from orchestrator.policy.errors import (
    ContractError, LeakageError, SchemaVersionError,
)


# -- allowlist invariants ----------------------------------------------------
#
# These assert properties of the column sets themselves rather than of any code
# path, because the sets are data and data is what a careless edit widens.


def test_no_decision_point_can_see_a_label():
    assert not (contract.D0_OBSERVABLE & contract.LABEL_COLUMNS)
    assert not (contract.D1_OBSERVABLE & contract.LABEL_COLUMNS)


def test_d0_is_a_strict_subset_of_d1():
    """D1 is D0 plus what generating revealed.

    If this ever fails, some column is visible before generating and invisible
    after, which is not a thing that can be true — it means an allowlist was
    edited by hand instead of by widening the D1-only set.
    """
    assert contract.D0_OBSERVABLE < contract.D1_OBSERVABLE


def test_sweep_wall_clock_is_never_a_feature():
    """`wall_ms` under `mode == "sweep"` is queue depth, not model speed."""
    assert "wall_ms" in contract.NEVER_A_FEATURE
    assert "wall_ms" not in contract.D0_OBSERVABLE
    assert "wall_ms" not in contract.D1_OBSERVABLE


def test_cost_columns_are_d1_only():
    """At D0 these are what the cost head predicts, never what it reads."""
    for column in ("gpu_seconds", "imputed_latency_s", "usd"):
        assert column in contract.D1_OBSERVABLE
        assert column not in contract.D0_OBSERVABLE


def test_visible_test_outcome_is_d1_only():
    """The whole reason the D0 to D1 comparison isolates information."""
    for column in ("visible_passed", "visible_total"):
        assert column in contract.D1_OBSERVABLE
        assert column not in contract.D0_OBSERVABLE


def test_observable_at_rejects_an_unknown_decision_point():
    with pytest.raises(ContractError, match="exactly"):
        contract.observable_at("D2")


# -- normalization -----------------------------------------------------------


def test_r1_row_normalizes_unchanged_in_the_fields_that_matter(raw_row):
    row = contract.normalize_row(raw_row())
    assert row["task_id"] == "mbpp/1"
    assert row["seed"] == 0
    assert row["prefill_tokens"] == 120
    assert row["code_parses"] is True
    assert row["schema_version"] == 1


def test_a_missing_required_column_names_itself(raw_row):
    row = raw_row()
    del row["params_hash"]
    with pytest.raises(ContractError, match="params_hash"):
        contract.normalize_row(row, where="part-0.jsonl:12")


def test_the_message_carries_where_it_happened(raw_row):
    row = raw_row(task_id="")
    with pytest.raises(ContractError, match=r"part-0\.jsonl:12"):
        contract.normalize_row(row, where="part-0.jsonl:12")


def test_an_unsupported_schema_version_is_refused(raw_row):
    with pytest.raises(SchemaVersionError, match="schema_version 2"):
        contract.normalize_row(raw_row(schema_version=2))


def test_a_fractional_count_is_refused_rather_than_rounded(raw_row):
    """Rounding a token count silently changes what it counts."""
    with pytest.raises(ContractError, match="prefill_tokens"):
        contract.normalize_row(raw_row(prefill_tokens=12.5))


def test_a_widened_integer_still_reads(raw_row):
    """A Parquet round-trip may hand back 120.0 for an int column."""
    row = contract.normalize_row(raw_row(prefill_tokens=120.0))
    assert row["prefill_tokens"] == 120


def test_nullable_columns_survive_as_none(raw_row):
    row = contract.normalize_row(raw_row())
    for column in contract.GRADE_COLUMNS:
        assert row[column] is None
    assert row["gpu_seconds"] is None
    assert row["parent_rollout_id"] is None


@pytest.mark.parametrize("value,expected", [
    (None, None),
    ([], ()),
    (["hardcoded"], ("hardcoded",)),
    ('["hardcoded", "skipped"]', ("hardcoded", "skipped")),
    ("hardcoded", ("hardcoded",)),
])
def test_hack_flags_normalize_from_every_shape_r2_might_write(
    raw_row, value, expected,
):
    row = contract.normalize_row(raw_row(hack_flags=value))
    assert row["hack_flags"] == expected


def test_extra_reads_the_same_as_a_dict_and_as_json(raw_row):
    """The guard against R3's two read paths diverging.

    R1 serializes `extra` to a JSON string when writing Parquet, because Arrow
    would otherwise infer a struct from the first row and fail on the rest. The
    same run read through JSONL and through Parquet must still be one run.
    """
    as_dict = contract.normalize_row(raw_row(extra={"note": "x", "n": 2}))
    as_json = contract.normalize_row(raw_row(extra='{"note": "x", "n": 2}'))
    assert as_dict["extra"] == as_json["extra"] == {"note": "x", "n": 2}


def test_unparseable_extra_is_kept_rather_than_dropped(raw_row):
    row = contract.normalize_row(raw_row(extra="not json"))
    assert row["extra"] == {"_unparsed": "not json"}


@pytest.mark.parametrize("value,expected", [
    (True, True), ("true", True), (1, True),
    (False, False), ("false", False), (0, False),
])
def test_booleans_survive_a_string_round_trip(raw_row, value, expected):
    row = contract.normalize_row(raw_row(code_parses=value))
    assert row["code_parses"] is expected


# -- graded or not -----------------------------------------------------------


def test_an_r1_row_is_not_graded(raw_row):
    """Today every real row is in exactly this state."""
    assert contract.is_graded(contract.normalize_row(raw_row())) is False


def test_a_graded_row_is_graded(graded_row):
    assert contract.is_graded(contract.normalize_row(graded_row())) is True


def test_a_row_with_no_hidden_tests_is_not_a_training_example(graded_row):
    row = contract.normalize_row(graded_row(hidden_passed=0, hidden_total=0))
    assert contract.is_graded(row) is False


def test_a_row_with_no_visible_tests_is_not_a_d1_example(graded_row):
    """A D1 decision is made on a visible-test outcome. No tests, no outcome."""
    row = contract.normalize_row(graded_row(visible_passed=None,
                                            visible_total=None))
    assert contract.is_graded(row) is False


@pytest.mark.parametrize("passed,total,expected", [
    (8, 8, True),
    (7, 8, False),
    (0, 8, False),
    (0, 0, False),
    (None, 8, False),
])
def test_solved_is_all_or_nothing(graded_row, passed, total, expected):
    row = contract.normalize_row(
        graded_row(hidden_passed=passed, hidden_total=total)
    )
    assert contract.solved(row) is expected


# -- the leak guard ----------------------------------------------------------


def test_assert_no_labels_names_every_offending_column():
    with pytest.raises(LeakageError) as excinfo:
        contract.assert_no_labels(
            ["task_id", "hidden_passed", "hidden_total"],
            context="D0 feature frame",
        )
    message = str(excinfo.value)
    assert "hidden_passed" in message
    assert "hidden_total" in message
    assert "D0 feature frame" in message


def test_assert_no_labels_passes_a_clean_column_set():
    contract.assert_no_labels(sorted(contract.D1_OBSERVABLE), context="D1")
