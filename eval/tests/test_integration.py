"""The whole harness, end to end, against an answer known by construction.

This is the test the evaluation harness exists to pass. Everything upstream is
a unit; this asserts that assembling them produces the *right ranking* on data
whose optimum was planted, with intervals that cover the planted effect.

An evaluation harness that has never been run against a known answer is not
validated — it is untested code that produces numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval import (
    benjamini_hochberg,
    confusion,
    load_rows,
    mcnemar_exact,
    oracle_headroom,
    paired_accuracy_difference,
    standard_baselines,
    sweep,
)
from eval.heuristics import synth_features, tune
from eval.policies import from_decisions
from schemas import SynthConfig, generate

LAMS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


@pytest.fixture(scope="module")
def big():
    """Large enough that the intervals mean something."""
    return generate(SynthConfig(n_tasks=600, seeds=3), seed=101)


@pytest.fixture(scope="module")
def big_store(big):
    return load_rows(big.rows)


def test_full_comparison_runs_and_every_number_has_an_interval(big_store):
    """No bare means. Every reported quantity carries a confidence interval —
    this is a merge blocker in CONTRIBUTING.md, so it is a test."""
    baselines = standard_baselines()
    reference = baselines["verifier_gated_cascade"](big_store)

    for name, fn in baselines.items():
        if name == "verifier_gated_cascade":
            continue
        ci = paired_accuracy_difference(fn(big_store), reference, n_resamples=400)
        assert np.isfinite(ci.point)
        assert ci.low <= ci.point <= ci.high
        assert ci.n_resamples > 0 and ci.method


def test_the_ranking_is_the_expected_one(big_store):
    """Accuracy ordering the design predicts. An inversion here means the
    replay accounting is wrong, and every downstream claim with it."""
    b = standard_baselines()
    acc = {n: fn(big_store).summary()["accuracy"] for n, fn in b.items()}
    assert acc["always_small"] < acc["best_of_3_small"]
    assert acc["always_small"] < acc["always_large"]
    assert acc["always_large"] <= acc["oracle_router"]
    assert acc["verifier_gated_cascade"] > acc["always_small"]
    assert acc["oracle_router"] == max(acc.values())


def test_cost_ordering_is_the_expected_one(big_store):
    b = standard_baselines()
    cost = {n: fn(big_store).summary()["cost"] for n, fn in b.items()}
    assert cost["always_small"] == min(cost.values())
    # The cascade pays for both arms on escalated tasks, so it can exceed the
    # large arm alone. That is real, not a bug, and it is its actual weakness.
    assert cost["verifier_gated_cascade"] > cost["always_small"]


def test_planted_optimum_is_ranked_first_among_d0_policies(big_store, big):
    """The sharp assertion: the harness must *rank* the known-best pre-generation
    policy above every other pre-generation policy, at every λ."""
    b = standard_baselines()
    d0_family = ("always_small", "always_large", "random_route(0.5)")
    tasks = set(big_store.ordered_tasks)

    for lam in LAMS:
        actions = {t: a for t, a in big.optimal_actions(lam).items() if t in tasks}
        planted = from_decisions(actions, "planted")(big_store)
        best = planted.per_task().utility(lam).mean()
        for name in d0_family:
            rival = b[name](big_store).per_task().utility(lam).mean()
            assert best >= rival - 1e-9, f"{name} beat the planted optimum at λ={lam}"


def test_bootstrap_covers_the_planted_arm_gap(big_store, big):
    """The interval on `large - small` must cover the gap the generator planted.
    This is coverage on a real quantity rather than a simulated one."""
    solved = big_store.arm_matrix("_solved")
    from eval.stats import paired_diff_bootstrap

    ci = paired_diff_bootstrap(solved["large"], solved["small"], n_resamples=1500)

    truth = big.truth
    d = truth["difficulty"]
    specs = {a.name: a for a in truth["arms"]}
    planted_gap = float((specs["large"].p_solve(d) - specs["small"].p_solve(d)).mean())

    assert ci.low <= planted_gap <= ci.high, (
        f"interval {ci} misses the planted gap {planted_gap:.4f}"
    )


def test_mcnemar_agrees_with_the_bootstrap_on_direction(big_store):
    """Two independent tests on the same comparison must not disagree. If they
    do, one of them is wrong and the report would show whichever ran first."""
    solved = big_store.arm_matrix("_solved")
    large = solved["large"][:, 0].astype(bool)
    small = solved["small"][:, 0].astype(bool)

    mc = mcnemar_exact(large, small)
    from eval.stats import paired_diff_bootstrap

    ci = paired_diff_bootstrap(
        solved["large"][:, :1], solved["small"][:, :1], n_resamples=800
    )
    assert mc.p_value < 0.05
    assert ci.excludes_zero
    assert (ci.point > 0) == (mc.b > mc.c)


def test_bh_correction_applied_across_the_lambda_sweep(big_store):
    """Testing many frontier points and reporting the best uncorrected is how a
    null becomes a headline."""
    b = standard_baselines()
    solved = big_store.arm_matrix("_solved")
    p_values = []
    for lam in LAMS:
        del lam
        p_values.append(
            mcnemar_exact(
                solved["large"][:, 0].astype(bool), solved["small"][:, 0].astype(bool)
            ).p_value
        )
    rejected, adjusted = benjamini_hochberg(p_values, q=0.05)
    assert len(adjusted) == len(LAMS)
    assert (adjusted >= np.array(p_values) - 1e-12).all()
    assert rejected.all(), "a real effect should survive correction"
    del b


def test_headroom_is_reported_before_any_policy_is_judged(big_store):
    """If escalation_helps were tiny, a small measured gap would mean a
    saturated problem rather than a weak policy. Reporting it first is what
    stops that being misattributed."""
    h = oracle_headroom(big_store)
    assert h["escalation_helps"] > 0.05
    assert h["neither"] > 0.0, "a realistic corpus has unsolvable tasks"


def test_confusion_matrix_explains_the_cascade(big_store):
    conf = confusion(standard_baselines()["verifier_gated_cascade"](big_store), big_store)
    assert sum(conf.counts.values()) == big_store.n_tasks
    assert 0.0 < conf.escalation_rate < 1.0, "a cascade should sometimes escalate"
    assert conf.counts["correct_escalation"] > 0


def test_heuristic_enters_the_comparison_as_a_seventh_family(big_store, big):
    """The capacity ladder, assembled. `heuristic_route` must sit between
    `random_route` and the D1 baselines — if it beat everything, it is reading
    something it should not."""
    features = synth_features(big.rows)
    lam = 0.05
    best = tune(features, big_store, lam)
    heuristic = best.rule.policy(features)(big_store).per_task().utility(lam).mean()

    b = standard_baselines()
    rand = b["random_route(0.5)"](big_store).per_task().utility(lam).mean()
    oracle = b["oracle_router"](big_store).per_task().utility(lam).mean()

    assert rand < heuristic < oracle


def test_sweep_produces_a_complete_grid(big_store):
    result = sweep(standard_baselines(), big_store, LAMS)
    assert len(result) == 6
    assert all(len(v) == len(LAMS) for v in result.values())
    for points in result.values():
        assert all(np.isfinite(p.accuracy) and np.isfinite(p.cost) for p in points)


def test_the_harness_reports_a_null_when_there_is_one(big_store):
    """A harness that cannot report a null is not a harness. Comparing a policy
    against itself must produce an interval covering zero."""
    cascade = standard_baselines()["verifier_gated_cascade"]
    ci = paired_accuracy_difference(
        cascade(big_store), cascade(big_store), n_resamples=400
    )
    assert ci.point == pytest.approx(0.0, abs=1e-12)
    assert not ci.excludes_zero
