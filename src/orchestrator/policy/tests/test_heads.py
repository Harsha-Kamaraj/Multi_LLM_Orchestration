"""The three heads, and the separations that make them worth having.

The load-bearing tests here are the ones about what is *not* in the model:
λ is not, dollars are not, and the arm is not. Each of those would produce a
policy that works on the day it is fitted and silently stops meaning what it
says the first time something outside R3's control changes.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from orchestrator.policy import fixtures, heads, store
from orchestrator.policy.calibration import Calibrator
from orchestrator.policy.errors import SplitError
from orchestrator.policy.features import feature_set
from orchestrator.policy.features.d0 import D0_FEATURES
from orchestrator.policy.heads import (
    ArmHeads, HeadError, PolicyHeads, fit_arm, fit_heads,
)
from orchestrator.workers.cost import CostCoefficients, LinearFit
from schemas.synth import SynthConfig

CONFIG = SynthConfig(n_tasks=300, seeds=3)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory) -> store.RolloutData:
    root = tmp_path_factory.mktemp("heads")
    fx = fixtures.write_fixture(root, CONFIG)
    return store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path,
                               cost_fingerprint=fx.cost_fingerprint)


@pytest.fixture(scope="module")
def policy(loaded) -> PolicyHeads:
    return fit_heads(loaded, "D1")


def _coefficients(decode_s: float = 0.01, usd_per_hour: float = 1.10,
                  intercept: float = 0.5) -> CostCoefficients:
    """A pinned costing, standing in for R1's characterization pass."""
    fit = LinearFit(intercept_s=intercept, prefill_s_per_token=4.0e-5,
                    decode_s_per_token=decode_s, n=72, r2=0.999, rmse_s=0.01)
    return CostCoefficients(
        models={"synthetic/small": {1: fit}, "synthetic/large": {1: fit}},
        hardware="test", usd_per_gpu_hour=usd_per_hour,
    )


# -- shape ------------------------------------------------------------------


def test_one_head_per_arm(policy, loaded):
    assert policy.arm_names == tuple(sorted(set(loaded.arms)))
    assert len(policy.arms) == 2


def test_a_policy_with_no_arms_is_refused(loaded):
    with pytest.raises(HeadError, match="cannot choose anything"):
        PolicyHeads(run_id="r", decision_point="D1",
                    features=feature_set("D1"), arms={})


def test_asking_for_an_arm_that_was_never_fitted_says_which_exist(policy, loaded):
    rows = [r for r in loaded.rows if r["split"] == "val"][:5]
    with pytest.raises(HeadError, match="no head for arm"):
        policy.p_pass("direct_enormous", rows)


# -- P_pass is a probability ------------------------------------------------


def test_p_pass_is_bounded(policy, loaded):
    rows = [r for r in loaded.rows
            if r["split"] == "val" and r["arm"] == "small"][:200]
    p = policy.p_pass("small", rows)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_p_pass_is_calibrated_by_default(policy, loaded):
    """The default has to be the safe one — an uncalibrated default would be
    read as a probability by every caller that did not think about it."""
    rows = [r for r in loaded.rows
            if r["split"] == "val" and r["arm"] == "small"][:50]
    head = policy.arms["small"]
    assert head.calibrator is not None
    np.testing.assert_allclose(
        policy.p_pass("small", rows),
        head.calibrator(policy.p_pass("small", rows, calibrated=False)),
    )


def test_an_uncalibrated_head_refuses_to_hand_back_a_probability(policy, loaded):
    rows = [r for r in loaded.rows if r["split"] == "val"][:5]
    naked = ArmHeads(
        arm="small", model_id="synthetic/small", decision_point="D1",
        feature_names=policy.arms["small"].feature_names,
        scaler=policy.arms["small"].scaler,
        pass_model=policy.arms["small"].pass_model,
        prefill_model=None, decode_model=None, calibrator=None,
    )
    with pytest.raises(HeadError, match="no calibrator"):
        naked.p_pass(rows, policy.features)


# -- the arm indexes the head, it is not a feature --------------------------


def test_the_arm_is_not_a_feature(policy):
    """Pooling the arms with a one-hot would force one weight vector to
    describe both models. Separate fits let them disagree about difficulty."""
    for head in policy.arms.values():
        assert "arm" not in head.feature_names
        assert not any(name.startswith("arm_") for name in head.feature_names)


def test_the_two_arms_learned_different_weights(policy):
    small = policy.arms["small"].pass_model.coef_.ravel()
    large = policy.arms["large"].pass_model.coef_.ravel()
    assert not np.allclose(small, large), (
        "identical coefficients means the arms were not fitted separately"
    )


# -- E_cost predicts tokens, and converts at scoring time --------------------


def test_token_predictions_are_never_negative(policy, loaded):
    """An unconstrained linear fit goes negative at the low end, and a negative
    cost reads to the decision rule as a reward for picking that arm."""
    rows = [r for r in loaded.rows if r["split"] == "val"][:300]
    prefill, decode = policy.arms["small"].tokens(rows, policy.features)
    assert prefill.min() >= 0.0 and decode.min() >= 0.0


def test_the_price_is_applied_at_scoring_time_not_baked_in(policy, loaded):
    """The architectural commitment, stated as a test.

    One fitted policy, two costings. If the dollar figure moved while the
    predicted tokens did not, the head is predicting tokens and the price is a
    parameter — which is the whole point. Had `E_cost` been trained against
    dollars, a new instance price would need a retrain and nothing would say so.
    """
    rows = [r for r in loaded.rows if r["split"] == "val"][:100]
    head = policy.arms["small"]

    cheap = head.e_usd(rows, policy.features, _coefficients(usd_per_hour=1.10))
    dear = head.e_usd(rows, policy.features, _coefficients(usd_per_hour=4.40))

    np.testing.assert_allclose(dear, cheap * 4.0, rtol=1e-9)
    before, after = (head.tokens(rows, policy.features) for _ in range(2))
    np.testing.assert_array_equal(before[1], after[1])


def test_latency_carries_the_fixed_cost_and_gpu_seconds_does_not(policy, loaded):
    """R1's distinction, preserved rather than re-derived.

    A user waits through the per-request intercept; the GPU is not occupied by
    it. Collapsing the two would charge the policy for queueing.
    """
    rows = [r for r in loaded.rows if r["split"] == "val"][:100]
    head = policy.arms["small"]
    coefficients = _coefficients(intercept=0.5)

    cost = head.e_cost(rows, policy.features, coefficients)
    latency = head.e_latency(rows, policy.features, coefficients)
    np.testing.assert_allclose(latency - cost, 0.5, rtol=1e-6)


def test_a_bigger_decode_coefficient_costs_more(policy, loaded):
    rows = [r for r in loaded.rows if r["split"] == "val"][:50]
    head = policy.arms["small"]
    slow = head.e_cost(rows, policy.features, _coefficients(decode_s=0.02))
    fast = head.e_cost(rows, policy.features, _coefficients(decode_s=0.01))
    assert (slow > fast).all()


def test_an_arm_without_token_columns_loses_only_its_cost_heads(loaded):
    """An ungraded or uncosted run should not lose its whole policy."""
    stripped = tuple(dict(row, prefill_tokens=None, decode_tokens=None)
                     for row in loaded.rows)
    data = replace(loaded, rows=stripped)
    head, _ = fit_arm(data, "small", "D1")
    assert head.prefill_model is None
    with pytest.raises(HeadError, match="no token heads"):
        head.tokens(stripped[:5], feature_set("D1"))


# -- refusals ---------------------------------------------------------------


def test_scoring_with_a_different_feature_set_is_refused(policy, loaded):
    """D0 columns against a D1-fitted head would misalign every coefficient."""
    rows = [r for r in loaded.rows if r["split"] == "val"][:20]
    with pytest.raises(HeadError, match="different feature set"):
        policy.arms["small"]._design(rows, D0_FEATURES)


def test_an_arm_serving_two_models_is_refused(loaded):
    """Its cost would be an average over two hardware profiles."""
    mixed = tuple(
        dict(row, model_id="synthetic/other")
        if row["arm"] == "small" and row["seed"] == 0 else dict(row)
        for row in loaded.rows
    )
    data = replace(loaded, rows=mixed)
    with pytest.raises(HeadError, match="more than one model"):
        fit_arm(data, "small", "D1")


def test_no_train_rows_is_a_split_error(loaded):
    only_val = tuple(dict(row, split="val") for row in loaded.rows)
    data = replace(loaded, rows=only_val)
    with pytest.raises(SplitError, match="no train rows"):
        fit_arm(data, "small", "D1")


# -- artifacts --------------------------------------------------------------


def test_saving_writes_the_whole_artifact_set(policy, tmp_path):
    directory = policy.save(tmp_path / "policy")
    for name in (heads.POLICY_PICKLE, heads.HEADS_MANIFEST,
                 heads.CALIBRATION_FILE, heads.FEATURE_SPEC_FILE):
        assert (directory / name).exists(), name


def test_the_manifest_is_readable_without_unpickling_anything(policy, tmp_path):
    """R4 audits artifacts. One that needs R3's code to open is not auditable."""
    directory = policy.save(tmp_path / "policy")
    payload = json.loads((directory / heads.HEADS_MANIFEST).read_text())
    assert payload["decision_point"] == "D1"
    assert payload["run_id"] == policy.run_id
    assert set(payload["arms"]) == set(policy.arm_names)
    assert "meets_calibration_target" in payload


def test_the_calibration_file_holds_one_map_per_arm(policy, tmp_path):
    directory = policy.save(tmp_path / "policy")
    payload = json.loads((directory / heads.CALIBRATION_FILE).read_text())
    assert set(payload) == set(policy.arm_names)
    restored = Calibrator.from_dict(payload["small"])
    assert restored == policy.arms["small"].calibrator


def test_a_reloaded_policy_scores_identically(policy, loaded, tmp_path):
    """A policy that predicts differently after a round trip is not an
    artifact, and every downstream number would be unreproducible."""
    directory = policy.save(tmp_path / "policy")
    reloaded = PolicyHeads.load(directory)
    rows = [r for r in loaded.rows if r["split"] == "val"][:100]
    np.testing.assert_allclose(
        reloaded.p_pass("small", rows), policy.p_pass("small", rows),
    )


def test_publishability_follows_the_run(policy, loaded):
    assert policy.publishable == loaded.publishable


def test_the_summary_names_which_map_shipped(policy):
    text = policy.summary()
    assert "arms:" in text
    assert "isotonic" in text or "identity" in text


# -- calibration, end to end ------------------------------------------------


def test_every_arm_is_calibrated(policy):
    assert set(policy.calibration) == set(policy.arm_names)
    for report in policy.calibration.values():
        assert report.n_rows > 0


def test_the_target_is_judged_per_arm_not_on_average(policy):
    """One badly calibrated arm corrupts every comparison it takes part in, so
    an average would hide exactly the failure that matters."""
    assert policy.meets_calibration_target == all(
        r.meets_target for r in policy.calibration.values()
    )


def test_d0_heads_read_only_d0_columns(loaded):
    """The separation the whole D0/D1 comparison rests on."""
    from orchestrator.policy import contract

    d0 = fit_heads(loaded, "D0")
    for feature in d0.features:
        for column in feature.source_columns:
            assert column in contract.D0_OBSERVABLE, column
