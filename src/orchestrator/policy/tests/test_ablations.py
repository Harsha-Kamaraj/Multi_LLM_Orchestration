"""Ablations, and the two ways an ablation table lies.

The first is multiplicity: run sixteen comparisons at 95% and one will exclude
zero by luck. The second is subtler and shows up constantly on fixtures — a
group whose features are *constant* ablates to exactly zero, which reads in a
table as "this group does not matter" and actually means "this store had no
variation to remove".
"""

from __future__ import annotations

import pytest

from orchestrator.policy import ablations, fixtures, store
from orchestrator.policy.ablations import (
    FEATURE_GROUPS, AblationError, check_groups_cover, group_of, only_group,
    run_ablations, without_group,
)
from orchestrator.policy.features import feature_set
from orchestrator.policy.features.d0 import D0_FEATURES
from orchestrator.policy.features.d1 import D1_FEATURES
from schemas.synth import SynthConfig

CONFIG = SynthConfig(n_tasks=500, seeds=3)
RESAMPLES = 200


@pytest.fixture(scope="module")
def loaded(tmp_path_factory) -> store.RolloutData:
    root = tmp_path_factory.mktemp("ablations")
    fx = fixtures.write_fixture(root, CONFIG)
    return store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path)


@pytest.fixture(scope="module")
def d1_table(loaded):
    return run_ablations(loaded, "D1", n_resamples=RESAMPLES)


# -- the groups themselves --------------------------------------------------


def test_every_feature_belongs_to_a_group():
    """An ungrouped feature is invisible to every leave-one-out, so it can
    carry an effect the table then attributes to nothing."""
    assert check_groups_cover(D0_FEATURES) == ()
    assert check_groups_cover(D1_FEATURES) == ()


def test_no_feature_is_in_two_groups():
    seen: dict[str, str] = {}
    for group, members in FEATURE_GROUPS.items():
        for name in members:
            assert name not in seen, f"{name} in {group} and {seen.get(name)}"
            seen[name] = group


def test_removing_a_group_removes_exactly_it(loaded):
    reduced = without_group(D1_FEATURES, "code_shape")
    assert all(group_of(f.name) != "code_shape" for f in reduced)
    assert len(list(reduced)) == len(list(D1_FEATURES)) - len(
        FEATURE_GROUPS["code_shape"]
    )


def test_only_a_group_keeps_exactly_it():
    kept = only_group(D1_FEATURES, "visible_outcome")
    assert {f.name for f in kept} == set(FEATURE_GROUPS["visible_outcome"])


def test_removing_a_group_that_is_not_there_is_refused():
    with pytest.raises(AblationError, match="removes nothing"):
        without_group(D0_FEATURES, "code_shape")


def test_keeping_only_an_absent_group_is_refused():
    with pytest.raises(AblationError, match="contributes no feature"):
        only_group(D0_FEATURES, "code_shape")


def test_an_unknown_group_is_refused(loaded):
    with pytest.raises(AblationError, match="unknown feature group"):
        run_ablations(loaded, "D0", groups=["not_a_group"],
                      n_resamples=RESAMPLES)


# -- the finding ------------------------------------------------------------


def test_the_visible_outcome_carries_the_d0_to_d1_gap(d1_table):
    """The ablation that turns the project's slogan into a claim.

    D1 adds three kinds of evidence at once. If the whole gap is the visible
    test outcome, "observing failure beats predicting it" is about
    verification. If it were code shape, the finding would be the much more
    ordinary "messier code is likelier to be wrong".
    """
    best = d1_table.carries_the_gap()
    assert best is not None
    assert best.group == "visible_outcome"
    assert best.delta.point < -0.1


def test_removing_the_visible_outcome_lands_near_the_d0_number(d1_table, loaded):
    """The strong form: D1 without verification is roughly D0."""
    without = next(d for d in d1_table.by_impact()
                   if d.group == "visible_outcome")
    d0 = run_ablations(loaded, "D0", n_resamples=RESAMPLES)
    assert abs(without.auc - d0.baseline_auc) < 0.08


def test_no_d0_group_is_individually_detectable(loaded):
    """A real answer, and the one a wide family usually gives: the D0 groups
    are redundant rather than useless."""
    table = run_ablations(loaded, "D0", n_resamples=RESAMPLES)
    assert table.carries_the_gap() is None


# -- multiplicity -----------------------------------------------------------


def test_the_intervals_are_simultaneous(d1_table):
    """Sixteen comparisons at 95% each is roughly a coin flip that one excludes
    zero by luck. Reporting that one is how a null becomes a headline."""
    assert d1_table.family_size > 1
    expected = 1.0 - (1.0 - d1_table.level) / d1_table.family_size
    assert d1_table.simultaneous_level == pytest.approx(expected)
    for delta in d1_table.deltas:
        assert delta.delta.level == pytest.approx(expected)


def test_correcting_makes_the_intervals_wider(loaded):
    """If the correction did not widen anything it would not be a correction."""
    one = run_ablations(loaded, "D1", groups=["prompt_shape"],
                        include_only=False, n_resamples=RESAMPLES)
    many = run_ablations(loaded, "D1", n_resamples=RESAMPLES)

    uncorrected = one.deltas[0].delta.width
    corrected = next(d for d in many.deltas
                     if d.group == "prompt_shape"
                     and d.kind == "leave_one_out").delta.width
    assert corrected > uncorrected


# -- the vacuous-zero guard -------------------------------------------------


def test_a_constant_group_is_flagged_rather_than_reported_as_useless(d1_table):
    """`schemas.synth` writes a near-identical stub for every `code`, so every
    code-shape feature is flat. Its ablation is vacuous, and "removal changed
    nothing" would otherwise read as a result about the group."""
    assert "code_shape" in d1_table.degenerate_groups()
    code = next(d for d in d1_table.by_impact() if d.group == "code_shape")
    assert code.degenerate
    assert code.delta.point == 0.0 and code.delta.width == 0.0


def test_a_group_that_genuinely_does_not_help_is_not_flagged(d1_table):
    """The distinction the guard exists for: a real but unhelpful group still
    moves under resampling, so its interval has width."""
    prompt = next(d for d in d1_table.by_impact() if d.group == "prompt_shape")
    assert not prompt.degenerate
    assert prompt.delta.width > 0.0
    assert not prompt.significant


def test_the_summary_marks_degenerate_and_significant_differently(d1_table):
    text = d1_table.summary()
    assert "* without visible_outcome" in text
    assert "? without code_shape" in text
    assert "constant on this run" in text


# -- shape ------------------------------------------------------------------


def test_both_kinds_of_variant_are_run(d1_table):
    kinds = {d.kind for d in d1_table.deltas}
    assert kinds == {"leave_one_out", "only"}


def test_only_variants_can_be_switched_off(loaded):
    table = run_ablations(loaded, "D1", include_only=False,
                          n_resamples=RESAMPLES)
    assert {d.kind for d in table.deltas} == {"leave_one_out"}


def test_the_table_serializes_with_every_interval(d1_table):
    payload = d1_table.as_dict()
    assert payload["carries_the_gap"] == "visible_outcome"
    assert payload["degenerate_groups"]
    for delta in payload["deltas"]:
        assert "low" in delta["delta"] and "high" in delta["delta"], (
            "no bare means: every number leaves with an interval"
        )


def test_a_run_without_both_splits_is_refused(loaded):
    from dataclasses import replace

    train_only = tuple(dict(r, split="train") for r in loaded.rows)
    with pytest.raises(AblationError, match="needs both train and val"):
        run_ablations(replace(loaded, rows=train_only), "D1",
                      n_resamples=RESAMPLES)
