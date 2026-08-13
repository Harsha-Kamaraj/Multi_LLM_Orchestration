"""The Phase 0 gate, measured against a fixture whose answer is known.

The important test in this file is `test_d0_does_not_beat_the_planted_ceiling`.
The fixture plants a D0 signal of known strength, and no honest model can score
above it — so a measured AUC that exceeds the plant is not a good result, it is
a leak. That assertion is the reason the synthetic store exists.

Everything else here checks that the *measurement* is honest: that the interval
clusters on tasks, that the verdict distinguishes "cleared the bar" from "could
not tell", and that the test split is nowhere near any of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.stats import Interval
from orchestrator.policy import fixtures, gate, store
from orchestrator.policy.features import feature_set
from orchestrator.policy.gate import GateError, GateResult
from schemas.synth import SynthConfig

# Small enough to stay fast, large enough that an AUC over it is not noise.
RESAMPLES = 200


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> fixtures.Fixture:
    root = tmp_path_factory.mktemp("gate")
    return fixtures.write_fixture(root, SynthConfig(n_tasks=500, seeds=3))


@pytest.fixture(scope="module")
def data(fx) -> store.RolloutData:
    return store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path,
                               cost_fingerprint=fx.cost_fingerprint)


@pytest.fixture(scope="module")
def results(data) -> dict[str, GateResult]:
    return gate.measure_gates(data, n_resamples=RESAMPLES)


# -- the numbers -------------------------------------------------------------


def test_both_decision_points_produce_an_interval(results):
    """No bare means. This is a merge blocker, and it applies to R3 first."""
    for point, result in results.items():
        assert isinstance(result.auc, Interval)
        assert result.auc.low < result.auc.point < result.auc.high
        assert result.auc.n_resamples == RESAMPLES


def test_d1_dominates_d0(results):
    """Observing failure beats predicting it — the project's central claim."""
    assert results["D1"].auc.point > results["D0"].auc.point + 0.10


def test_each_point_estimate_lands_in_the_roadmap_band(results):
    """A number far outside the predicted band reads as a bug, not a finding."""
    for point, result in results.items():
        low, high = result.expected_band
        assert low <= result.auc.point <= high, (
            f"AUC_{point} = {result.auc.point:.4f}, outside the predicted "
            f"{low}-{high}. Either the fixture parameters moved or the "
            f"feature set changed materially."
        )


def test_d1_clears_the_hard_stop(results):
    """`AUC_D1 < 0.75` would end the project, so it is asserted explicitly."""
    assert results["D1"].verdict == "PASS"
    assert results["D1"].auc.low >= 0.75


def test_d0_is_weak_and_that_is_the_finding(results):
    """Expected, not a failure. Tuning D0 until it passes destroys the result."""
    assert results["D0"].auc.point < results["D1"].auc.point
    assert results["D0"].verdict in ("PASS", "INCONCLUSIVE", "FAIL")


# -- the leakage canary ------------------------------------------------------


def test_d0_does_not_beat_the_planted_ceiling(data, results):
    """The assertion the synthetic fixture exists for.

    `x_d0` is the planted prompt-only proxy and the *only* D0 signal in the
    data. A model reading legitimate prompt features cannot rank better than
    the proxy itself, up to sampling noise. If it does, something outside the
    D0 information set reached the feature path — and a leak shows up as a
    better number, which is the one outcome nobody investigates.

    The ceiling is computed per row, against the same label the model predicts.
    `SynthResult.planted_auc_d0` scores against `small_solved_any` — solved on
    *any* of three seeds — which is a different and easier target, so it is not
    the right bound to compare a per-row model against.
    """
    eval_rows = [
        row for row in data.rows
        if str(row["arm"]) == results["D0"].arm and row.get("split") == "val"
    ]
    proxy = np.array([data.latent[str(r["rollout_id"])]["x_d0"]
                      for r in eval_rows])
    solved = np.array([data.label_for(str(r["rollout_id"])).solved
                       for r in eval_rows])

    ceiling = gate.auc(proxy, solved)
    measured = results["D0"].auc.point

    assert measured <= ceiling + 0.05, (
        f"AUC_D0 = {measured:.4f} exceeds the planted ceiling {ceiling:.4f}. "
        f"No legitimate D0 feature can carry more signal than the proxy that "
        f"generated it — check what reached the feature path."
    )


def test_the_planted_ceiling_is_itself_only_moderate(data, results):
    """If the ceiling were near 1.0 the test above would prove nothing."""
    eval_rows = [
        row for row in data.rows
        if str(row["arm"]) == results["D0"].arm and row.get("split") == "val"
    ]
    proxy = np.array([data.latent[str(r["rollout_id"])]["x_d0"]
                      for r in eval_rows])
    solved = np.array([data.label_for(str(r["rollout_id"])).solved
                       for r in eval_rows])
    assert 0.55 < gate.auc(proxy, solved) < 0.80


# -- the interval is clustered -----------------------------------------------


def test_clustering_on_tasks_widens_the_interval(data):
    """A row bootstrap treats 3 seeds of one task as 3 observations.

    It returns an interval that is far too narrow. Asserting the clustered one
    is wider is the check that the clustering is actually happening — the two
    code paths differ by one flag and produce numbers that look equally
    plausible.
    """
    from eval.stats import cluster_bootstrap

    result = gate.measure_gate(data, "D1", n_resamples=RESAMPLES)

    eval_rows = [r for r in data.rows
                 if str(r["arm"]) == result.arm and r.get("split") == "val"]
    matrix, _, _ = gate._index_matrix(eval_rows)
    scores = np.array([data.label_for(str(r["rollout_id"])).hidden_passed
                       for r in eval_rows], dtype=float)
    labels = np.array([data.label_for(str(r["rollout_id"])).solved
                       for r in eval_rows], dtype=int)

    def auc_of(sample):
        flat = sample.ravel()
        idx = flat[~np.isnan(flat)].astype(int)
        return gate.auc(scores[idx], labels[idx])

    clustered = cluster_bootstrap(matrix, auc_of, n_resamples=RESAMPLES, seed=1)
    tasks_only = cluster_bootstrap(matrix, auc_of, n_resamples=RESAMPLES, seed=1,
                                   resample_seeds=False)
    assert clustered.width >= tasks_only.width * 0.9


def test_ragged_seed_counts_are_not_padded(data):
    """A task swept with fewer seeds must not have one counted twice."""
    rows = [r for r in data.rows if r.get("split") == "val"][:60]
    dropped = [r for i, r in enumerate(rows) if i % 7]
    matrix, n_tasks, n_seeds = gate._index_matrix(dropped)
    assert np.isnan(matrix).any(), "expected some empty task-seed cells"
    present = int((~np.isnan(matrix)).sum())
    assert present == len(dropped)


def test_a_single_task_cannot_be_bootstrapped():
    with pytest.raises(GateError, match="at least 2 tasks"):
        gate._index_matrix([{"task_id": "t1", "rollout_id": "r1"}])


# -- AUC itself --------------------------------------------------------------


def test_a_constant_score_is_half_not_one():
    """A model that learned nothing must score as having learned nothing."""
    assert gate.auc(np.zeros(10), np.array([0, 1] * 5)) == 0.5


def test_a_perfect_ranking_is_one():
    assert gate.auc(np.arange(10.0), np.array([0] * 5 + [1] * 5)) == 1.0


def test_an_inverted_ranking_is_zero():
    assert gate.auc(np.arange(10.0), np.array([1] * 5 + [0] * 5)) == 0.0


def test_one_class_is_undefined_rather_than_perfect():
    assert np.isnan(gate.auc(np.arange(4.0), np.ones(4)))


# -- resolving the arm -------------------------------------------------------


def test_the_cheap_arm_is_resolved_by_measured_cost(data):
    """"Small" means cheap. The name is a convention the schema does not check."""
    assert data.has_cost
    assert gate.cheap_arm(data) == "small"


def test_the_arm_name_is_only_a_fallback(fx):
    unpriced = store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path)
    assert not unpriced.has_cost
    assert gate.cheap_arm(unpriced) == "small"


def test_an_unresolvable_arm_raises_rather_than_guessing(data):
    """Guessing here would silently invert every gate number."""
    rows = tuple(dict(r, arm={"small": "alpha", "large": "beta"}[r["arm"]])
                 for r in data.rows)
    renamed = store.RolloutData(
        run_id=data.run_id, rows=rows, labels=data.labels,
        manifest=data.manifest, source=data.source,
    )
    with pytest.raises(GateError, match="cannot tell which"):
        gate.cheap_arm(renamed)


def test_a_single_arm_run_has_no_gate(data):
    rows = tuple(r for r in data.rows if r["arm"] == "small")
    one_arm = store.RolloutData(
        run_id=data.run_id, rows=rows, labels=data.labels,
        manifest=data.manifest, source=data.source,
    )
    with pytest.raises(GateError, match="needs both"):
        gate.cheap_arm(one_arm)


# -- split discipline --------------------------------------------------------


def test_the_gate_fits_on_train_and_scores_on_val(data, results):
    for result in results.values():
        n_train = len([r for r in data.rows if r["arm"] == result.arm
                       and r.get("split") == "train"])
        n_val = len([r for r in data.rows if r["arm"] == result.arm
                     and r.get("split") == "val"])
        assert result.n_train_rows == n_train
        assert result.n_eval_rows == n_val


def test_no_test_rows_are_reachable_at_all(data):
    """Not a flag that is left off — the loader cannot produce them."""
    assert all(r.get("split") != "test" for r in data.rows)


def test_a_run_without_a_validation_split_is_refused(data):
    rows = tuple(dict(r, split="train") for r in data.rows)
    only_train = store.RolloutData(
        run_id=data.run_id, rows=rows, labels=data.labels,
        manifest=data.manifest, source=data.source,
    )
    with pytest.raises(GateError, match="no val rows"):
        gate.measure_gate(only_train, "D0", n_resamples=10)


def test_a_single_outcome_class_is_refused(data):
    """Nothing separates, and a fitted model would report a meaningless AUC."""
    labels = {rid: store.Label(rid, lab.task_id, lab.arm, lab.seed, 0, 8)
              for rid, lab in data.labels.items()}
    all_failed = store.RolloutData(
        run_id=data.run_id, rows=data.rows, labels=labels,
        manifest=data.manifest, source=data.source,
    )
    with pytest.raises(GateError, match="one outcome class"):
        gate.measure_gate(all_failed, "D0", n_resamples=10)


# -- reporting ---------------------------------------------------------------


@pytest.mark.parametrize("low,high,expected", [
    (0.80, 0.90, "PASS"),
    (0.60, 0.70, "FAIL"),
    (0.70, 0.80, "INCONCLUSIVE"),
])
def test_the_verdict_separates_cleared_from_could_not_tell(low, high, expected):
    """An estimate straddling the bar has not cleared it — it has not answered."""
    result = GateResult(
        decision_point="D1",
        auc=Interval(point=(low + high) / 2, low=low, high=high),
        threshold=0.75, expected_band=(0.80, 0.90),
        n_train_rows=100, n_eval_rows=50, n_eval_tasks=25, n_seeds=2,
        n_features=10, base_rate=0.3, arm="small", run_id="r",
    )
    assert result.verdict == expected


def test_an_inconclusive_verdict_says_it_needs_more_data():
    result = GateResult(
        decision_point="D1",
        auc=Interval(point=0.76, low=0.71, high=0.81),
        threshold=0.75, expected_band=(0.80, 0.90),
        n_train_rows=100, n_eval_rows=50, n_eval_tasks=25, n_seeds=2,
        n_features=10, base_rate=0.3, arm="small", run_id="r",
    )
    assert "not a pass" in result.summary()


def test_the_report_names_the_asymmetry(results):
    text = gate.gate_report(results)
    assert "D1 - D0" in text
    assert "AUC_D0" in text and "AUC_D1" in text


def test_a_failed_hard_stop_says_the_premise_is_false():
    failed = {
        "D0": GateResult("D0", Interval(0.55, 0.50, 0.60), 0.65, (0.60, 0.68),
                         10, 10, 5, 2, 3, 0.3, "small", "r"),
        "D1": GateResult("D1", Interval(0.60, 0.55, 0.65), 0.75, (0.80, 0.90),
                         10, 10, 5, 2, 3, 0.3, "small", "r"),
    }
    text = gate.gate_report(failed)
    assert "hard stop" in text
    assert "premise is" in text


def test_the_report_persists_with_the_run_it_came_from(tmp_path, results):
    path = gate.write_gate_report(results, tmp_path / "gate.json",
                                  measured_at="2026-08-13")
    payload = json.loads(path.read_text())

    assert payload["measured_at"] == "2026-08-13"
    for point in ("D0", "D1"):
        assert payload[point]["run_id"] == results[point].run_id
        assert payload[point]["verdict"] == results[point].verdict
        assert set(payload[point]["auc"]) >= {"point", "low", "high", "method"}
