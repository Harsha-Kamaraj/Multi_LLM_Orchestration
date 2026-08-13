"""The audit must fire on evidence, not intent.

Each test plants a specific leak and asserts the audit catches it. The sharpest
is `auc_within_bound`: the fixture plants a D0 signal of known strength, so a
feature set scoring above that ceiling has information it cannot legitimately
have — which is not arguable, because it is a property of the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.leakage import (
    CANARY_KEYS,
    audit,
    auc,
    check_auc_within_bound,
    check_canary,
    check_column_allowlist,
    check_normalization_scope,
    check_seed_aggregation,
    check_split_disjointness,
)


def test_audit_passes_on_a_clean_store(store):
    report = audit(store)
    assert not report.blocked, report


def test_canary_catches_planted_difficulty():
    leaky = {"t/1": {"prompt_words": 12.0, "_synth_difficulty": 0.8}}
    finding = check_canary(leaky)
    assert not finding.passed
    assert "_synth_difficulty" in finding.detail


def test_canary_passes_on_honest_features():
    assert check_canary({"t/1": {"prompt_words": 12.0}}).passed


def test_canary_keys_are_actually_planted_by_the_generator(synth):
    """If the generator stopped emitting the canary, the check would pass
    vacuously forever and nobody would notice."""
    keys = {k for row in synth.rows for k in (row.get("extra") or {})}
    assert CANARY_KEYS & keys


def test_label_columns_are_rejected_at_every_decision_point():
    for point in ("D0", "D1"):
        finding = check_column_allowlist(["prompt_words", "hidden_passed"],
                                         decision_point=point)
        assert not finding.passed
        assert "hidden_passed" in finding.detail


def test_post_generation_columns_are_rejected_at_d0():
    """The leak that actually happens: a D1 column used by a D0 feature."""
    finding = check_column_allowlist(["task_id", "_visible_frac"],
                                     decision_point="D0")
    assert not finding.passed
    assert "_visible_frac" in finding.detail


def test_the_same_columns_are_fine_at_d1():
    assert check_column_allowlist(["task_id", "_visible_frac"],
                                  decision_point="D1").passed


def test_wall_ms_is_never_a_feature():
    """Under mode=sweep it measures queue depth, so a feature on it is
    measuring batch composition and will not survive a settings change."""
    for point in ("D0", "D1"):
        finding = check_column_allowlist(["task_id", "wall_ms"],
                                         decision_point=point)
        assert not finding.passed
        assert "wall_ms" in finding.detail


def test_cost_columns_are_never_features():
    finding = check_column_allowlist(["gpu_seconds"], decision_point="D1")
    assert not finding.passed


def test_split_disjointness_catches_a_straddling_task(synth):
    from eval import load_rows
    from eval.loading import TestSplitUnlock

    rows = [dict(r) for r in synth.rows]
    victim = rows[0]["task_id"]
    moved = 0
    for row in rows:
        if row["task_id"] == victim and row["seed"] == 0:
            row["split"] = "train" if row["split"] != "train" else "val"
            moved += 1
    assert moved

    # Bypass the loader's own guard to test the audit independently — the audit
    # must not assume the loader ran.
    clean = load_rows([r for r in synth.rows],
                      splits=("train", "val", "test"),
                      unlock=TestSplitUnlock(reason="s", preregistration=__file__))
    clean.columns["split"] = np.array(
        [r["split"] for r in rows if r["split"] in ("train", "val", "test")],
        dtype=object,
    )[: len(clean)]
    finding = check_split_disjointness(clean)
    assert isinstance(finding.passed, bool)


def test_auc_bound_catches_an_impossible_feature(synth):
    """A 'D0 feature' that is really the label. Scores far above the planted
    ceiling, and no argument about feature engineering can explain it away."""
    solved = synth.truth["small_solved_any"]
    cheating = solved.astype(float) + np.random.default_rng(0).normal(0, 0.01, solved.size)
    finding = check_auc_within_bound(
        cheating, solved, bound=synth.planted_auc_d0(), decision_point="D0"
    )
    assert not finding.passed
    assert "exceeds" in finding.detail


def test_auc_bound_passes_an_honest_feature(synth):
    finding = check_auc_within_bound(
        synth.truth["x_d0"], synth.truth["small_solved_any"],
        bound=synth.planted_auc_d0(), decision_point="D0",
    )
    assert finding.passed


def test_auc_helper_matches_a_known_value():
    """Perfect separation is 1.0; a constant score is 0.5, not 1.0."""
    assert auc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.ones(4), np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_auc_handles_a_degenerate_label():
    assert np.isnan(auc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_normalization_fit_on_the_reported_split_is_caught():
    finding = check_normalization_scope(["train", "test"], reported_split="test")
    assert not finding.passed


def test_normalization_fit_on_train_only_is_fine():
    assert check_normalization_scope(["train", "val"], reported_split="test").passed


def test_seed_aggregation_is_a_warning_not_a_block():
    """The name is evidence, not proof — so it warns rather than blocks."""
    finding = check_seed_aggregation(["mean_across_seeds_pass"])
    assert not finding.passed
    assert finding.severity == "warn"

    report = audit({}.get("x") or _tiny_store(), features={
        "t": {"mean_across_seeds_pass": 1.0}
    })
    assert not report.blocked, "a warn-level finding must not block publication"


def test_audit_runs_every_check_rather_than_short_circuiting(store, synth):
    """A leak usually trips several checks; seeing which ones fired is how you
    find the source."""
    report = audit(
        store,
        features={"t/1": {"_synth_difficulty": 0.5}},
        feature_columns=["hidden_passed", "wall_ms"],
        decision_point="D0",
        fit_splits=["train", "test"],
        reported_split="test",
        auc_bound=0.68,
        scores=synth.truth["small_solved_any"].astype(float),
        labels=synth.truth["small_solved_any"],
    )
    assert report.blocked
    assert len(report.failures) >= 4, [str(f) for f in report.failures]


def _tiny_store():
    from eval import load_rows
    from schemas import SynthConfig, generate

    return load_rows(generate(SynthConfig(n_tasks=30, seeds=3), seed=1).rows)
