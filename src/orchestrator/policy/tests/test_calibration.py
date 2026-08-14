"""Calibration, and the ways a calibration number lies.

Most of these are negative tests. A calibrated and an uncalibrated `P_pass`
produce identical rankings, identical AUC, and identical-looking plots — the
difference only shows up once λ starts multiplying one of them against a cost,
which is far too late to find out.
"""

from __future__ import annotations

import numpy as np
import pytest

from orchestrator.policy.calibration import (
    ECE_TARGET, Calibrator, CalibrationError, CalibrationReport,
    brier_score, calibrate, cross_fitted_probabilities,
    expected_calibration_error, fit_calibrator, maximum_calibration_error,
    reliability_table,
)


def _perfect(n: int = 4000, seed: int = 0):
    """Probabilities that are true by construction, so ECE must be ~0."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.random(n) < p).astype(int)
    return p, y


def _squashed(n: int = 900, seed: int = 1):
    """A ranking that is right and probabilities that are wrong.

    Scores are compressed toward 0.5, which is what an over-regularized
    classifier produces: the order is preserved, so AUC is untouched, and every
    probability is understated at the top and overstated at the bottom.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    true_p = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.random(n) < true_p).astype(int)
    p = 0.5 + 0.25 * (true_p - 0.5)
    groups = [f"task_{i // 3}" for i in range(n)]
    return p, y, groups


# -- the measure ------------------------------------------------------------


def test_a_truthful_predictor_scores_near_zero():
    p, y = _perfect()
    assert expected_calibration_error(p, y) < 0.02


def test_a_squashed_predictor_is_caught():
    p, y, _ = _squashed()
    assert expected_calibration_error(p, y) > ECE_TARGET


def test_ece_alone_is_fooled_by_a_predictor_that_ignores_its_input():
    """Why Brier is reported beside it.

    Predicting the base rate for every row is perfectly calibrated and entirely
    useless. ECE says it is excellent; Brier does not move. Reporting ECE on
    its own would let a model that learned nothing claim the target.
    """
    _, y, _ = _squashed()
    base = np.full(y.shape, y.mean())
    assert expected_calibration_error(base, y) < 0.01
    assert brier_score(base, y) > 0.2


def test_the_worst_bin_can_be_bad_while_the_mean_looks_fine():
    """One sparse, confident, wrong bin is invisible to a count-weighted mean."""
    y = np.zeros(1000, dtype=int)
    p = np.zeros(1000)
    y[:500] = 1
    p[:500] = 0.5          # 500 rows, perfectly calibrated
    p[500:990] = 0.5
    p[990:] = 0.99         # 10 rows, confidently wrong
    assert expected_calibration_error(p, y) < 0.02
    assert maximum_calibration_error(p, y) > 0.9


def test_empty_bins_are_dropped_rather_than_reported_as_zero():
    """A bin nothing landed in is not evidence that the model is wrong there."""
    p = np.full(200, 0.42)
    y = (np.arange(200) % 2).astype(int)
    table = reliability_table(p, y)
    assert len(table) == 1
    assert table[0].n == 200


def test_every_row_lands_in_exactly_one_bin():
    p, y = _perfect(n=500)
    assert sum(b.n for b in reliability_table(p, y)) == 500


def test_a_probability_of_exactly_one_is_not_its_own_bin():
    p = np.array([1.0, 1.0, 0.95, 0.05])
    y = np.array([1, 1, 1, 0])
    assert len(reliability_table(p, y)) == 2


def test_a_score_outside_zero_one_is_refused():
    """A decision-function output is not a probability, and looks like one."""
    with pytest.raises(CalibrationError, match=r"must lie in \[0, 1\]"):
        expected_calibration_error(np.array([-1.2, 0.5]), np.array([0, 1]))


def test_mismatched_lengths_are_refused():
    with pytest.raises(CalibrationError, match="same rows in the same order"):
        expected_calibration_error(np.array([0.1, 0.2]), np.array([1]))


def test_nan_probabilities_are_refused():
    with pytest.raises(CalibrationError, match="NaN"):
        expected_calibration_error(np.array([0.5, np.nan]), np.array([0, 1]))


# -- the fitted map ---------------------------------------------------------


def test_calibration_never_inverts_the_ranking():
    """The map is monotone, so no pair of rows swaps order. Ties are the only
    change it may introduce, and it introduces a lot of them: isotonic is a
    step function, so 900 distinct scores become a couple of dozen levels.

    Stated as order preservation rather than as "AUC is unchanged", because
    AUC is *not* unchanged — merging ranks moves a tie-averaged AUC, and an
    isotonic map fitted on these same labels moves it upward. That is in-sample
    optimism rather than a better ranking, which is the same reason the
    headline ECE is cross-fitted.
    """
    p, y, _ = _squashed()
    calibrated = fit_calibrator(p, y)(p)

    order = np.argsort(p, kind="mergesort")
    along = calibrated[order]
    assert np.all(np.diff(along) >= -1e-12), "a monotone map inverted a pair"
    assert len(set(along.tolist())) < len(set(p.tolist()))


def test_fitting_on_test_is_refused():
    p, y, _ = _squashed()
    with pytest.raises(CalibrationError, match="refusing to fit"):
        fit_calibrator(p, y, fitted_on=("val", "test"))


def test_a_calibrator_that_claims_test_cannot_be_constructed():
    """The refusal is on the object, not only on the fitting function."""
    with pytest.raises(CalibrationError, match="not a calibrator"):
        Calibrator(knots_x=(0.0, 1.0), knots_y=(0.0, 1.0),
                   fitted_on=("test",), n_rows=10)


def test_one_outcome_class_is_refused():
    p = np.linspace(0.1, 0.9, 100)
    with pytest.raises(CalibrationError, match="one outcome class"):
        fit_calibrator(p, np.ones(100, dtype=int))


def test_a_constant_score_is_refused():
    y = (np.arange(100) % 2).astype(int)
    with pytest.raises(CalibrationError, match="no ranking to calibrate"):
        fit_calibrator(np.full(100, 0.5), y)


def test_a_non_monotone_map_is_refused():
    with pytest.raises(CalibrationError, match="non-decreasing"):
        Calibrator(knots_x=(0.0, 0.5, 1.0), knots_y=(0.0, 0.9, 0.4),
                   fitted_on=("val",), n_rows=10)


def test_applying_it_clamps_outside_the_fitted_range():
    p, y, _ = _squashed()
    calibrator = fit_calibrator(p, y)
    out = calibrator(np.array([0.0, 1.0]))
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_it_refuses_to_calibrate_something_that_is_not_a_probability():
    p, y, _ = _squashed()
    with pytest.raises(CalibrationError, match="outside"):
        fit_calibrator(p, y)(np.array([2.0]))


def test_a_calibrator_round_trips_through_disk(tmp_path):
    """It ships beside the heads, so scoring reloads it and never refits."""
    p, y, _ = _squashed()
    calibrator = fit_calibrator(p, y, arm="direct_small")
    reloaded = Calibrator.load(calibrator.save(tmp_path / "calibration.json"))
    assert reloaded == calibrator
    np.testing.assert_allclose(reloaded(p), calibrator(p))


def test_the_artifact_is_readable_without_importing_r3(tmp_path):
    """R4 audits artifacts. One that needs R3's code to open is not auditable."""
    import json

    p, y, _ = _squashed()
    path = fit_calibrator(p, y, arm="direct_small").save(tmp_path / "c.json")
    payload = json.loads(path.read_text())
    assert payload["arm"] == "direct_small"
    assert payload["fitted_on"] == ["val"]
    manual = np.interp([0.5], payload["knots_x"], payload["knots_y"])
    np.testing.assert_allclose(manual, Calibrator.load(path)(np.array([0.5])))


# -- the honest number ------------------------------------------------------


def test_in_sample_ece_flatters_the_result():
    """The reason the headline number is cross-fitted.

    Isotonic is flexible enough to drive in-sample ECE to essentially zero on
    any data at all. If this test ever fails, the cross-fitting has stopped
    doing anything and the reported number has quietly become the optimistic
    one.
    """
    p, y, groups = _squashed()
    _, report = calibrate(p, y, groups)
    assert report.ece_after < 0.01
    assert report.ece_cross_fitted > report.ece_after
    assert report.optimism > 0


def test_calibration_improves_on_the_uncalibrated_probabilities():
    p, y, groups = _squashed()
    _, report = calibrate(p, y, groups)
    assert report.ece_cross_fitted < report.ece_before


def test_the_verdict_uses_the_cross_fitted_number():
    p, y, groups = _squashed()
    _, report = calibrate(p, y, groups)
    assert report.meets_target == (report.ece_cross_fitted < ECE_TARGET)


def test_seeds_of_one_task_never_straddle_a_fold():
    """The whole point of grouping. Without it the honest number is not honest.

    Constructed so each task's three seeds share a label: if a fold split them,
    the held-out row would be calibrated by a map that had already seen its
    answer, and cross-fitted ECE would collapse toward the in-sample value.
    """
    rng = np.random.default_rng(3)
    n_tasks = 120
    labels, probs, groups = [], [], []
    for t in range(n_tasks):
        outcome = int(rng.random() < 0.5)
        score = rng.uniform(0.4, 0.6)
        for _ in range(3):
            labels.append(outcome)
            probs.append(score)
            groups.append(f"task_{t}")
    p = np.array(probs)
    y = np.array(labels)

    cross = cross_fitted_probabilities(p, y, groups, n_folds=4, seed=0)
    # Every seed of a task shares a score, so a leak would make each task's
    # calibrated value equal its own outcome. It must not.
    by_task = {g: cross[i] for i, g in enumerate(groups)}
    assert not np.allclose(sorted(by_task.values()), sorted(
        {g: y[i] for i, g in enumerate(groups)}.values()))


def test_cross_fitting_needs_more_tasks_than_folds():
    p, y, _ = _squashed(n=30)
    with pytest.raises(CalibrationError, match="cannot be split"):
        cross_fitted_probabilities(p, y, [f"t{i}" for i in range(30)],
                                   n_folds=50)


def test_a_group_per_row_is_refused_when_it_does_not_match():
    p, y, groups = _squashed()
    with pytest.raises(CalibrationError, match="one per row"):
        cross_fitted_probabilities(p, y, groups[:-1])


def test_cross_fitting_is_deterministic():
    p, y, groups = _squashed()
    a = cross_fitted_probabilities(p, y, groups, seed=7)
    b = cross_fitted_probabilities(p, y, groups, seed=7)
    np.testing.assert_array_equal(a, b)


def test_the_report_serializes_with_its_bins():
    p, y, groups = _squashed()
    _, report = calibrate(p, y, groups, arm="direct_small")
    payload = report.as_dict()
    assert payload["arm"] == "direct_small"
    assert payload["ece_target"] == ECE_TARGET
    assert payload["bins"] and "observed_rate" in payload["bins"][0]
    assert isinstance(report, CalibrationReport)
    assert "cross-fitted" in report.summary()
