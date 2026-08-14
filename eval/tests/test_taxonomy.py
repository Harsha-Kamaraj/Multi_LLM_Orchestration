"""The taxonomy must partition, and must rank severity the right way.

Two properties carry everything: exactly one category per row (so the columns
sum to 100% and a reader can trust them), and `reward_hack` outranking success
(so a hacked run is never reported as a good one).
"""

from __future__ import annotations

import pytest

from eval.taxonomy import (
    CATEGORIES,
    MEANING,
    classify,
    classify_row,
    classify_store,
    explain_gap,
)


def _row(**kw):
    base = {
        "arm": "small", "hidden_passed": 0, "hidden_total": 8,
        "visible_passed": 0, "visible_total": 4,
        "error_class": "none", "finish_reason": "stop", "hack_flags": [],
    }
    base.update(kw)
    return base


# --- severity ordering -----------------------------------------------------

def test_reward_hack_outranks_success():
    """A generation that passed by defeating the harness is a measurement
    failure, not a success with a caveat. Burying it under `solved` is how a
    hacked run gets reported as a good one."""
    category = classify_row(
        solved=1.0, visible_frac=1.0, error_class="none",
        finish_reason="stop", hack_flags=["hardcoded_visible_case"],
    )
    assert category == "reward_hack"


def test_harness_error_outranks_a_code_failure():
    """The grader failed, not the code. Counting it as a capability failure
    blames the model for infrastructure."""
    assert classify_row(
        solved=0.0, visible_frac=0.0, error_class="harness_error",
        finish_reason="stop", hack_flags=[],
    ) == "harness_error"


def test_truncation_is_checked_after_success():
    """A generation that hit the token limit and still passed every hidden test
    is solved. Calling it a failure would understate the arm."""
    assert classify_row(
        solved=1.0, visible_frac=1.0, error_class="none",
        finish_reason="length", hack_flags=[],
    ) == "solved"
    assert classify_row(
        solved=0.0, visible_frac=0.0, error_class="none",
        finish_reason="length", hack_flags=[],
    ) == "truncated"


def test_overfit_visible_is_distinguished_from_wrong_answer():
    """Passing every visible test and failing hidden ones is the failure a
    cascade structurally cannot catch — its escalation signal is exactly the
    visible tests this row satisfied."""
    assert classify_row(
        solved=0.0, visible_frac=1.0, error_class="none",
        finish_reason="stop", hack_flags=[],
    ) == "overfit_visible"
    assert classify_row(
        solved=0.0, visible_frac=0.25, error_class="none",
        finish_reason="stop", hack_flags=[],
    ) == "wrong_answer"


@pytest.mark.parametrize("error_class,expected", [
    ("timeout", "timeout"),
    ("empty_code", "empty_code"),
    ("syntax_error", "syntax_error"),
    ("runtime_error", "runtime_error"),
])
def test_real_grader_error_classes_map_to_categories(error_class, expected):
    """The vocabulary comes from R2's grader, not from an invented one. If R2
    adds a class, this test is where the gap shows up."""
    assert classify_row(
        solved=0.0, visible_frac=0.0, error_class=error_class,
        finish_reason="stop", hack_flags=[],
    ) == expected


# --- partitioning ----------------------------------------------------------

def test_every_row_gets_exactly_one_category():
    rows = [
        _row(), _row(hidden_passed=8), _row(error_class="timeout"),
        _row(hack_flags=["reads_test_file"]), _row(finish_reason="length"),
        _row(visible_passed=4), _row(error_class="syntax_error"),
    ]
    tax = classify(rows)
    assert sum(tax.counts.values()) == tax.n == len(rows)


def test_rates_sum_to_one():
    """A taxonomy whose columns sum to more than 100% teaches readers to
    distrust it."""
    rows = [_row(hidden_passed=8 if i % 3 == 0 else 0) for i in range(60)]
    tax = classify(rows)
    assert sum(tax.rate(c) for c in CATEGORIES) == pytest.approx(1.0)


def test_every_category_has_a_meaning():
    """A bare count makes a reader reconstruct the reasoning. The report
    carries the interpretation with the number."""
    assert set(MEANING) == set(CATEGORIES)
    assert all(MEANING[c].strip() for c in CATEGORIES)


def test_failure_rate_is_the_complement_of_solved():
    rows = [_row(hidden_passed=8) for _ in range(7)] + [_row() for _ in range(3)]
    tax = classify(rows)
    assert tax.rate("solved") == pytest.approx(0.7)
    assert tax.failure_rate == pytest.approx(0.3)


def test_dominant_failure_ignores_success():
    rows = [_row(hidden_passed=8) for _ in range(50)] + \
           [_row(error_class="timeout") for _ in range(5)] + \
           [_row() for _ in range(2)]
    assert classify(rows).dominant_failure() == "timeout"


def test_dominant_failure_is_none_when_everything_solves():
    assert classify([_row(hidden_passed=8) for _ in range(5)]).dominant_failure() is None


def test_zero_hidden_tests_is_not_solved_by_vacuous_truth():
    """`0 == 0` would make a task with no hidden tests count as solved."""
    assert classify_row(
        solved=0.0, visible_frac=0.0, error_class="none",
        finish_reason="stop", hack_flags=[],
    ) != "solved"
    tax = classify([_row(hidden_passed=0, hidden_total=0)])
    assert tax.counts["solved"] == 0


# --- per-arm attribution ---------------------------------------------------

def test_per_arm_counts_partition_within_each_arm():
    rows = [_row(arm="small") for _ in range(10)] + \
           [_row(arm="large", hidden_passed=8) for _ in range(10)]
    tax = classify(rows)
    assert sum(tax.by_arm["small"].values()) == 10
    assert sum(tax.by_arm["large"].values()) == 10


def test_explain_gap_attributes_a_difference_to_a_mechanism():
    """Turns 'the large arm is 19 points better' into a claim about why —
    which is the difference between a benchmark number and a finding."""
    rows = [_row(arm="small", error_class="timeout") for _ in range(20)] + \
           [_row(arm="large", hidden_passed=8) for _ in range(20)]
    lines = explain_gap(classify(rows))
    assert any("timeout" in line for line in lines)
    assert any("solved" in line for line in lines)


def test_explain_gap_is_empty_without_both_arms():
    assert explain_gap(classify([_row(arm="small")])) == []


def test_explain_gap_skips_negligible_differences():
    rows = [_row(arm="small", hidden_passed=8) for _ in range(100)] + \
           [_row(arm="large", hidden_passed=8) for _ in range(100)]
    assert explain_gap(classify(rows)) == []


# --- against the real fixture ----------------------------------------------

def test_classify_store_runs_on_a_loaded_store(store):
    tax = classify_store(store)
    assert tax.n == len(store)
    assert sum(tax.counts.values()) == tax.n
    assert set(tax.by_arm) == set(store.arms)


def test_the_fixture_shows_the_large_arm_solving_more(store):
    tax = classify_store(store)
    assert tax.arm_delta("solved", "large", "small") > 0.08


def test_table_renders_without_empty_categories(store):
    text = str(classify_store(store))
    assert "category" in text and "solved" in text
    assert "harness_error" not in text, "a zero-count category should not print"
