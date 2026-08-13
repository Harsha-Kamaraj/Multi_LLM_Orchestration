"""Baselines must behave the way the report claims they behave.

The assertions here are mostly *orderings* — floor below ceiling, cascade
expensive but accurate, oracle above everything. An ordering that inverts is a
bug; an ordering that holds is weak evidence the accounting is right. The
sharper checks are the ones about cost accounting, because that is where a
baseline gets quietly flattered.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.policies import (
    always,
    best_of_n_small,
    from_decisions,
    oracle_router,
    random_route,
    standard_baselines,
    verifier_gated_cascade,
)


def test_all_standard_baselines_run(store):
    outcomes = {name: fn(store) for name, fn in standard_baselines().items()}
    assert len(outcomes) == 6
    for name, out in outcomes.items():
        s = out.summary()
        assert 0.0 <= s["accuracy"] <= 1.0, name
        assert s["cost"] > 0, name


def test_large_beats_small_by_the_planted_gap(store):
    small = always("small")(store).summary()
    large = always("large")(store).summary()
    gap = large["accuracy"] - small["accuracy"]
    assert gap > 0.08, f"arm gap {gap:.3f} below the Phase 0 gate"
    assert large["cost"] > small["cost"]


def test_oracle_is_the_ceiling(store):
    """If any baseline beat the oracle, the oracle is computed wrongly and every
    headroom claim in the report is wrong with it."""
    oracle = oracle_router()(store).summary()["accuracy"]
    for name, fn in standard_baselines().items():
        if name == "oracle_router":
            continue
        assert fn(store).summary()["accuracy"] <= oracle + 1e-9, name


def test_oracle_never_escalates_on_hopeless_tasks(store):
    """Escalating where neither arm solves would spend money for nothing and
    make the ceiling unreachable for reasons unrelated to routing."""
    out = oracle_router()(store)
    grids = {a: store.arm_matrix("_solved")[a] for a in store.arms}
    hopeless = (grids["small"] < 1) & (grids["large"] < 1)
    assert (out.action[hopeless] == "small").all()


def test_cascade_pays_for_both_arms_when_it_escalates(store):
    """Serial escalation. Charging only the large arm would make the strongest
    baseline look stronger still."""
    out = verifier_gated_cascade()(store)
    small_cost = store.arm_matrix("gpu_seconds")["small"]
    escalated = np.char.find(out.action.astype(str), "->") >= 0
    assert (out.cost[escalated] > small_cost[escalated]).all()
    assert np.allclose(out.cost[~escalated], small_cost[~escalated])


def test_cascade_latency_is_additive_not_max(store):
    """The cascade's real weakness is p95, and it only appears if latency is
    summed. Taking the max would hide the tail the router competes on."""
    out = verifier_gated_cascade()(store)
    lat = store.arm_matrix("imputed_latency_s")
    escalated = np.char.find(out.action.astype(str), "->") >= 0
    expected = lat["small"][escalated] + lat["large"][escalated]
    assert np.allclose(out.latency[escalated], expected)
    assert (out.latency[escalated] > np.maximum(
        lat["small"][escalated], lat["large"][escalated])).all()


def test_cascade_p95_exceeds_its_mean_substantially(store):
    s = verifier_gated_cascade()(store).summary()
    assert s["latency_p95"] > s["latency_mean"] * 1.3


def test_cascade_beats_small_and_costs_more(store):
    """It is the baseline to beat precisely because it is good."""
    cascade = verifier_gated_cascade()(store).summary()
    small = always("small")(store).summary()
    assert cascade["accuracy"] > small["accuracy"]
    assert cascade["cost"] > small["cost"]


def test_best_of_n_charges_for_every_sample_drawn(store):
    """Samples are drawn sequentially; failed ones cannot be un-spent."""
    out = best_of_n_small(3)(store)
    one = always("small")(store)
    assert out.cost.mean() > one.cost.mean()
    assert out.consumes_all_seeds
    assert out.solved.shape[1] == 1, "consuming 3 seeds yields 1 replicate, not 3"


def test_best_of_n_helps_accuracy(store):
    """The confound this baseline exists to control for. If sampling did not
    help here, the control would be vacuous."""
    assert (best_of_n_small(3)(store).summary()["accuracy"]
            > always("small")(store).summary()["accuracy"])


def test_random_route_lands_between_the_fixed_arms(store):
    out = random_route(0.5)(store).summary()
    small = always("small")(store).summary()
    large = always("large")(store).summary()
    assert small["accuracy"] < out["accuracy"] < large["accuracy"]
    assert small["cost"] < out["cost"] < large["cost"]


def test_random_route_is_reproducible(store):
    a = random_route(0.3, seed=9)(store).summary()
    b = random_route(0.3, seed=9)(store).summary()
    assert a == b


def _planted(store, synth, lam):
    tasks = set(store.ordered_tasks)
    actions = {t: a for t, a in synth.optimal_actions(lam).items() if t in tasks}
    return from_decisions(actions, "planted_optimal")(store)


def test_planted_optimum_beats_every_d0_baseline(store, synth):
    """The generator knows the exact best action per task from latent
    difficulty, so it is the ceiling *among policies that decide before
    generating*. Beating the D0 family is the sharpest check available on the
    replay accounting.
    """
    lam = 0.05
    planted = _planted(store, synth, lam).per_task().utility(lam).mean()
    d0_family = ("always_small", "always_large", "random_route(0.5)")
    for name in d0_family:
        rival = standard_baselines()[name](store).per_task().utility(lam).mean()
        assert planted > rival, f"{name} beat a perfect D0 oracle"


def test_a_d1_baseline_can_beat_a_perfect_d0_oracle(store, synth):
    """The project's central claim, reproduced in the fixture.

    `best_of_n_small` observes visible-test outcomes; the planted optimum only
    knows latent difficulty and must commit before generating. Observing beats
    predicting — so a *perfect* pre-generation router loses to a cheap
    post-generation one.

    This is asserted rather than merely noted because it is the reason the
    architecture routes at D1. If it ever stops holding on this fixture, the
    fixture no longer models the problem the project is solving.
    """
    lam = 0.05
    planted = _planted(store, synth, lam).per_task().utility(lam).mean()
    d1 = standard_baselines()["best_of_3_small"](store).per_task().utility(lam).mean()
    assert d1 > planted, "the D0/D1 asymmetry has vanished from the fixture"


def test_oracle_router_remains_the_true_ceiling(store, synth):
    """The hidden-outcome oracle sees everything, including what the planted
    D0 optimum cannot. It must sit above both families."""
    lam = 0.05
    oracle = oracle_router()(store).per_task().utility(lam).mean()
    planted = _planted(store, synth, lam).per_task().utility(lam).mean()
    d1 = standard_baselines()["best_of_3_small"](store).per_task().utility(lam).mean()
    assert oracle > planted and oracle > d1


def test_from_decisions_refuses_to_default_a_missing_task(store):
    """Silently defaulting would attribute the default's outcome to the policy."""
    partial = {t: "small" for t in store.ordered_tasks[:-1]}
    with pytest.raises(KeyError, match="no action for"):
        from_decisions(partial, "partial")(store)


def test_unknown_arm_is_an_error(store):
    with pytest.raises(KeyError, match="not in store"):
        always("medium")(store)


def test_outcome_shapes_agree():
    from eval.policies import Outcome

    with pytest.raises(ValueError, match="disagree"):
        Outcome(
            name="broken",
            solved=np.zeros((3, 2)),
            cost=np.zeros((3, 3)),
            latency=np.zeros((3, 2)),
            action=np.zeros((3, 2), dtype=object),
        )
