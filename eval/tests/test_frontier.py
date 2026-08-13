"""Frontier construction, and the matched-cost discipline.

The tests that matter here are the refusals: no extrapolation past a measured
endpoint, and no comparison outside the overlapping cost range. Both are how a
losing policy gets reported as a winner.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.confusion import confusion, oracle_headroom
from eval.frontier import (
    Point,
    accuracy_at_cost,
    compare_at_matched_cost,
    evaluate,
    frontier_dominates,
    paired_accuracy_difference,
    pareto_front,
    sweep,
)
from eval.policies import always, standard_baselines, verifier_gated_cascade

LAMS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def test_sweep_covers_every_policy_and_lambda(store):
    result = sweep(standard_baselines(), store, LAMS)
    assert set(result) == set(standard_baselines())
    assert all(len(points) == len(LAMS) for points in result.values())


def test_fixed_arms_are_flat_in_cost_and_accuracy(store):
    """A fixed arm costs what it costs. Only its utility moves with λ — which
    is what shows the region where a fixed arm is genuinely optimal."""
    points = sweep({"always_small": always("small")}, store, LAMS)["always_small"]
    assert len({round(p.accuracy, 12) for p in points}) == 1
    assert len({round(p.cost, 12) for p in points}) == 1
    assert len({round(p.utility, 12) for p in points}) == len(LAMS)


def test_pareto_front_excludes_dominated_points():
    points = [
        Point("a", 0.0, accuracy=0.50, cost=1.0, latency_mean=1, latency_p95=1),
        Point("b", 0.0, accuracy=0.45, cost=2.0, latency_mean=1, latency_p95=1),
        Point("c", 0.0, accuracy=0.70, cost=3.0, latency_mean=1, latency_p95=1),
    ]
    front = [p.policy for p in pareto_front(points)]
    assert front == ["a", "c"], "b costs more for less accuracy"


def test_pareto_ties_break_toward_lower_cost():
    points = [
        Point("cheap", 0.0, accuracy=0.6, cost=1.0, latency_mean=1, latency_p95=1),
        Point("dear", 0.0, accuracy=0.6, cost=5.0, latency_mean=1, latency_p95=1),
    ]
    assert [p.policy for p in pareto_front(points)] == ["cheap"]


def test_accuracy_at_cost_interpolates():
    points = [
        Point("p", 0.0, accuracy=0.4, cost=1.0, latency_mean=1, latency_p95=1),
        Point("p", 0.0, accuracy=0.8, cost=3.0, latency_mean=1, latency_p95=1),
    ]
    assert accuracy_at_cost(points, 2.0) == pytest.approx(0.6)


def test_accuracy_at_cost_refuses_to_extrapolate():
    """Extrapolating past a measured endpoint invents a capability the policy
    was never shown to have — and it is exactly where a losing policy would
    like to be compared."""
    points = [
        Point("p", 0.0, accuracy=0.4, cost=1.0, latency_mean=1, latency_p95=1),
        Point("p", 0.0, accuracy=0.8, cost=3.0, latency_mean=1, latency_p95=1),
    ]
    assert accuracy_at_cost(points, 0.5) is None
    assert accuracy_at_cost(points, 9.0) is None


def test_matched_cost_comparison_uses_only_the_overlap():
    a = [
        Point("a", 0.0, accuracy=0.3, cost=1.0, latency_mean=1, latency_p95=1),
        Point("a", 0.0, accuracy=0.7, cost=5.0, latency_mean=1, latency_p95=1),
    ]
    b = [
        Point("b", 0.0, accuracy=0.5, cost=4.0, latency_mean=1, latency_p95=1),
        Point("b", 0.0, accuracy=0.9, cost=9.0, latency_mean=1, latency_p95=1),
    ]
    matched = compare_at_matched_cost(a, b)
    assert matched, "costs 4..5 overlap"
    assert all(4.0 - 1e-9 <= m.cost <= 5.0 + 1e-9 for m in matched)


def test_matched_cost_returns_nothing_without_overlap():
    a = [
        Point("a", 0.0, accuracy=0.3, cost=1.0, latency_mean=1, latency_p95=1),
        Point("a", 0.0, accuracy=0.4, cost=2.0, latency_mean=1, latency_p95=1),
    ]
    b = [
        Point("b", 0.0, accuracy=0.8, cost=8.0, latency_mean=1, latency_p95=1),
        Point("b", 0.0, accuracy=0.9, cost=9.0, latency_mean=1, latency_p95=1),
    ]
    assert compare_at_matched_cost(a, b) == []


def test_dominance_is_strict_and_usually_false(store):
    """The claim is not 'wins somewhere' — any two crossing curves achieve that
    by accident. This answers the stronger question and should often be False."""
    baselines = standard_baselines()
    small = sweep({"s": baselines["always_small"]}, store, LAMS)["s"]
    cascade = sweep({"c": baselines["verifier_gated_cascade"]}, store, LAMS)["c"]
    assert not frontier_dominates(small, cascade)


def test_paired_difference_handles_mismatched_replicates(store):
    """`best_of_n_small` yields one replicate; the fixed arms yield three.
    Comparing them must reduce to per-task means rather than raising."""
    a = standard_baselines()["best_of_3_small"](store)
    b = always("small")(store)
    ci = paired_accuracy_difference(a, b, n_resamples=400)
    assert ci.excludes_zero


def test_cascade_beats_small_with_an_interval_excluding_zero(store):
    ci = paired_accuracy_difference(
        verifier_gated_cascade()(store), always("small")(store), n_resamples=600
    )
    assert ci.point > 0 and ci.excludes_zero


# --- confusion matrix ------------------------------------------------------

def test_confusion_partitions_every_task(store):
    conf = confusion(always("small")(store), store)
    assert sum(conf.counts.values()) == conf.n == store.n_tasks


def test_always_small_has_no_escalations(store):
    conf = confusion(always("small")(store), store)
    assert conf.counts["false_escalation"] == 0
    assert conf.counts["correct_escalation"] == 0
    assert conf.escalation_rate == 0.0
    assert conf.counts["missed_escalation"] > 0


def test_always_large_never_misses_an_escalation(store):
    conf = confusion(always("large")(store), store)
    assert conf.counts["missed_escalation"] == 0
    assert conf.counts["correct_small"] == 0
    assert conf.counts["false_escalation"] > 0, "paying for large where small sufficed"


def test_unsolvable_is_excluded_from_regret(store):
    """A router cannot be blamed for tasks no arm solves. Folding them in makes
    every router look worse, uniformly — which also compresses the differences
    between routers."""
    conf = confusion(always("large")(store), store)
    assert conf.routable == conf.n - conf.counts["unsolvable"]
    assert conf.routable < conf.n, "the fixture should contain unsolvable tasks"
    assert 0.0 <= conf.regret <= 1.0


def test_always_large_is_warned_about(store):
    conf = confusion(always("large")(store), store)
    assert any("always_large" in w for w in conf.warnings)


def test_oracle_headroom_sums_to_one(store):
    h = oracle_headroom(store)
    total = h["small_only"] + h["large_only"] + h["both"] + h["neither"]
    assert total == pytest.approx(1.0)


def test_oracle_headroom_reports_real_escalation_value(store):
    """If this were near zero, no router could win by much and a small measured
    gap would mean a saturated problem rather than a weak policy."""
    h = oracle_headroom(store)
    assert h["escalation_helps"] > 0.05


def test_evaluate_reports_p95_above_mean(store):
    point = evaluate(verifier_gated_cascade()(store), lam=0.05)
    assert point.latency_p95 > point.latency_mean
    assert point.utility == pytest.approx(point.accuracy - 0.05 * point.cost)
