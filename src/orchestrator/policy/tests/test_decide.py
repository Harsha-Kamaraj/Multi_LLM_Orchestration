"""The decision rule and the λ sweep.

The test that matters most is the last one: R3's decisions replayed through
R4's `from_decisions` against R4's loader. Everything else here can be right
while the handoff is broken, and the handoff is the deliverable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from orchestrator.policy import decide, fixtures, heads, store
from orchestrator.policy.decide import (
    LAMBDA_GRID, ArmScore, DecisionError, Sweep, choose, read_actions,
    sweep_lambda, write_decisions,
)
from orchestrator.workers.cost import CostCoefficients, LinearFit
from schemas.synth import SynthConfig

CONFIG = SynthConfig(n_tasks=400, seeds=3)


def _coefficients() -> CostCoefficients:
    """A cheap arm and a genuinely dearer one, so λ has something to trade."""
    small = LinearFit(intercept_s=0.3, prefill_s_per_token=2.0e-5,
                      decode_s_per_token=0.004, n=72, r2=0.99, rmse_s=0.01)
    large = LinearFit(intercept_s=0.9, prefill_s_per_token=8.0e-5,
                      decode_s_per_token=0.020, n=72, r2=0.99, rmse_s=0.01)
    return CostCoefficients(
        models={"synthetic/small": {1: small}, "synthetic/large": {1: large}},
        hardware="test", usd_per_gpu_hour=1.10,
    )


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return fixtures.write_fixture(tmp_path_factory.mktemp("decide"), CONFIG)


@pytest.fixture(scope="module")
def loaded(fx) -> store.RolloutData:
    return store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path)


@pytest.fixture(scope="module")
def policy(loaded) -> heads.PolicyHeads:
    return heads.fit_heads(loaded, "D0")


@pytest.fixture(scope="module")
def sweep(policy, loaded) -> Sweep:
    return sweep_lambda(policy, loaded, _coefficients(), split="val")


# -- the rule ---------------------------------------------------------------


def test_a_tie_goes_to_the_cheaper_arm(loaded):
    """Ties are not hypothetical: at large λ cost dominates and equal-cost arms
    tie exactly. Breaking toward cheap keeps the sweep deterministic."""
    tied = [ArmScore("large", 0.5, 2.0, 2.0), ArmScore("small", 0.5, 1.0, 1.0)]
    assert choose(tied, lam=0.0).arm == "small"


def test_with_no_cost_pressure_the_best_predicted_arm_wins():
    scores = [ArmScore("large", 0.9, 5.0, 5.0), ArmScore("small", 0.2, 0.1, 0.1)]
    assert choose(scores, lam=0.0).arm == "large"


def test_with_enough_cost_pressure_the_cheapest_arm_wins():
    scores = [ArmScore("large", 0.9, 5.0, 5.0), ArmScore("small", 0.2, 0.1, 0.1)]
    assert choose(scores, lam=100.0).arm == "small"


def test_choosing_between_no_arms_is_refused():
    with pytest.raises(DecisionError, match="no arms"):
        choose([], lam=1.0)


# -- the sweep --------------------------------------------------------------


def test_the_grid_is_frozen_and_log_spaced():
    """A grid chosen after seeing the frontier is a knob. This one is written
    down, and a widening is a reviewed change to the constant."""
    assert len(LAMBDA_GRID) == 121
    assert LAMBDA_GRID[0] == pytest.approx(1e-4)
    assert LAMBDA_GRID[-1] == pytest.approx(1e2)
    ratios = np.diff(np.log10(np.array(LAMBDA_GRID)))
    np.testing.assert_allclose(ratios, ratios[0])


def test_routing_to_the_expensive_arm_is_monotone_in_lambda(sweep):
    """The invariant the frontier rests on.

    An arm is chosen over a cheaper one iff the quality gap beats λ times the
    cost gap. Raising λ raises the bar, so the set routed to the expensive arm
    can only shrink. If this ever fails, the rule is not the rule.
    """
    shares = [sweep.arm_share(lam)["large"] for lam in sweep.lambdas]
    assert all(a >= b - 1e-12 for a, b in zip(shares, shares[1:]))


def test_the_frontier_contains_both_degenerate_ends(sweep):
    """A sweep whose extremes are not degenerate has been cropped."""
    assert sweep.arm_share(sweep.lambdas[0])["large"] == 1.0
    assert sweep.arm_share(sweep.lambdas[-1])["small"] == 1.0


def test_a_degenerate_lambda_is_detected_exactly(sweep):
    """Regression: shares accumulated as `1/n` per task leave a fully
    degenerate split at 0.9999999999999999, and every exact comparison against
    1.0 silently reports the frontier as healthy at both ends."""
    assert sweep.lambdas[0] in sweep.degenerate_lambdas()
    assert sweep.lambdas[-1] in sweep.degenerate_lambdas()
    assert sweep.arm_share(sweep.lambdas[0])["large"] == 1.0


def test_some_lambda_routes_tasks_both_ways(sweep):
    """Without this the sweep is two constants and there is no frontier."""
    assert sweep.mixed_lambdas()


def test_every_task_gets_an_action_at_every_lambda(sweep):
    """R4 raises rather than defaulting a missing task, and it is right to."""
    for lam in sweep.lambdas:
        assert set(sweep.actions(lam)) == set(sweep.tasks)


def test_an_unswept_lambda_is_refused(sweep):
    """Interpolating would report a policy that was never run."""
    with pytest.raises(DecisionError, match="not on the frozen grid"):
        sweep.actions(0.0417)


def test_an_empty_grid_is_refused(policy, loaded):
    with pytest.raises(DecisionError, match="no frontier"):
        sweep_lambda(policy, loaded, _coefficients(), lambdas=[], split="val")


def test_every_record_pins_the_run(sweep, loaded):
    assert {r["run_id"] for r in sweep.records} == {loaded.run_id}


def test_publishability_follows_the_run(sweep, loaded):
    assert sweep.publishable == loaded.publishable


# -- what it refuses --------------------------------------------------------


def test_a_d1_policy_is_refused_with_the_accounting_reason(loaded):
    """The contract gap, stated as a refusal.

    Replaying a D1 escalation as "the action was large" charges for the large
    arm alone and drops the small arm's cost, which was already spent. That
    understates the policy and flatters it against every cascade baseline.
    """
    d1 = heads.fit_heads(loaded, "D1")
    with pytest.raises(DecisionError, match="already spent"):
        decide.score_tasks(d1, loaded, _coefficients())


def test_a_split_with_no_rows_says_so(policy, loaded):
    with pytest.raises(DecisionError, match="no rows"):
        sweep_lambda(policy, loaded, _coefficients(), split="nonexistent")


def test_a_task_swept_on_only_one_arm_cannot_be_routed(policy, loaded):
    from dataclasses import replace

    one_armed = tuple(r for r in loaded.rows if str(r["arm"]) == "small")
    with pytest.raises(DecisionError, match="no task was swept on every arm"):
        decide.score_tasks(policy, replace(loaded, rows=one_armed),
                           _coefficients())


# -- artifacts --------------------------------------------------------------


def test_writing_produces_jsonl_and_a_manifest(sweep, tmp_path):
    directory = write_decisions(sweep, tmp_path / "decisions")
    assert (directory / decide.DECISIONS_JSONL).exists()
    assert (directory / "decisions_manifest.json").exists()


def test_parquet_is_written_when_pyarrow_is_available(sweep, tmp_path):
    pytest.importorskip("pyarrow")
    directory = write_decisions(sweep, tmp_path / "decisions")
    assert (directory / decide.DECISIONS_PARQUET).exists()


def test_the_manifest_records_the_frozen_grid(sweep, tmp_path):
    directory = write_decisions(sweep, tmp_path / "decisions")
    payload = json.loads((directory / "decisions_manifest.json").read_text())
    assert payload["run_id"] == sweep.run_id
    assert payload["lambdas"] == list(sweep.lambdas)
    assert payload["degenerate_lambdas"]


def test_actions_round_trip_through_the_file(sweep, tmp_path):
    directory = write_decisions(sweep, tmp_path / "decisions")
    lam = sweep.mixed_lambdas()[0]
    from_disk = read_actions(directory / decide.DECISIONS_JSONL, lam)
    assert from_disk == sweep.actions(lam)


def test_reading_an_unswept_lambda_says_what_is_there(sweep, tmp_path):
    directory = write_decisions(sweep, tmp_path / "decisions")
    with pytest.raises(DecisionError, match="no decisions at lambda"):
        read_actions(directory / decide.DECISIONS_JSONL, 0.0417)


# -- the handoff ------------------------------------------------------------


def test_r4_can_replay_these_decisions(sweep, fx):
    """The deliverable, checked against R4's code rather than against a guess.

    `from_decisions` is the documented entry point for R3's `decisions.parquet`,
    and it raises unless every task in its store has an action. Loading R4's
    view of the same run and replaying through it is the only test that proves
    the handoff works.
    """
    from eval.loading import load_run
    from eval.policies import from_decisions

    r4 = load_run(fx.root, fx.run_id, splits=("val",))
    lam = sweep.mixed_lambdas()[0]
    outcome = from_decisions(sweep.actions(lam), f"learned_D0(lam={lam:.4g})")(r4)
    result = outcome.summary()

    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["cost"] > 0.0


def test_the_degenerate_ends_reproduce_r4s_constant_baselines(sweep, fx):
    """A strong check on the accounting: at the extremes the learned policy is
    `always_large` and `always_small`, so its numbers must match theirs exactly.
    Any disagreement is a bug in the decision layer, not a modelling result."""
    from eval.loading import load_run
    from eval.policies import always, from_decisions

    r4 = load_run(fx.root, fx.run_id, splits=("val",))
    for lam, arm in ((sweep.lambdas[0], "large"), (sweep.lambdas[-1], "small")):
        mine = from_decisions(sweep.actions(lam), "mine")(r4).summary()
        theirs = always(arm)(r4).summary()
        assert mine["accuracy"] == pytest.approx(theirs["accuracy"])
        assert mine["cost"] == pytest.approx(theirs["cost"])


# -- the charged probe -------------------------------------------------------


def test_a_probe_costs_k_cheap_samples(policy, loaded):
    """`PROBE_FEATURES` read k cheap draws, so using them obliges paying for
    them. A feature that assumes three free generations makes the policy look
    cheaper than any system that could actually run it."""
    scored = decide.score_tasks(policy, loaded, _coefficients(), split="val")
    surcharge = decide.probe_surcharge(policy, scored, cheap="small", k=3)

    task = next(iter(scored))
    cheap = next(s for s in scored[task] if s.arm == "small")
    assert surcharge[task] == pytest.approx(3 * cheap.e_cost)


def test_the_probe_charge_does_not_move_the_argmax(policy, loaded):
    """It is paid before the routing decision and whatever it says the money is
    gone, so it is a constant across arms — it shifts the policy rightward on
    the frontier rather than changing which arm wins at a fixed λ. The probe
    has to earn its cost through better decisions, not through free
    information."""
    scored = decide.score_tasks(policy, loaded, _coefficients(), split="val")
    surcharge = decide.probe_surcharge(policy, scored, cheap="small", k=3)

    for task, scores in scored.items():
        charged = tuple(
            decide.ArmScore(s.arm, s.p_pass, s.e_cost + surcharge[task],
                            s.e_latency)
            for s in scores
        )
        for lam in (0.001, 0.05, 10.0):
            assert (decide.choose(scores, lam).arm
                    == decide.choose(charged, lam).arm)


def test_a_policy_without_probe_features_is_charged_nothing(sweep):
    assert sweep.paid_arms == ()
    assert sweep.probe_cost_per_task == 0.0
    assert all(r["probe_cost"] == 0.0 for r in sweep.records)


def test_pricing_a_probe_needs_the_cheap_arm(policy, loaded):
    scored = decide.score_tasks(policy, loaded, _coefficients(), split="val")
    with pytest.raises(DecisionError, match="no 'probe_only' arm"):
        decide.probe_surcharge(policy, scored, cheap="probe_only")


def test_probe_features_cannot_be_attached_at_d0():
    """The gap this phase surfaced rather than closed.

    A self-consistency probe is bought *before* routing, so economically it
    belongs to a D0 decision — but its features read the outcomes of
    generations that have already happened, which are D1 columns. The contract
    has two decision points and this is a third surface. Recorded as a test so
    it is not rediscovered.
    """
    from orchestrator.policy.features import FeatureError, feature_set

    with pytest.raises(FeatureError, match="the probe features are D1"):
        feature_set("D0", with_probe=True)
