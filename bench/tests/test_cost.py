"""The cost model, tested against coefficients that are known by construction."""

from __future__ import annotations

import random

import pytest

from orchestrator.workers.cost import (
    CostCoefficients, LinearFit, Observation, fit_linear, validate_imputation,
)
from orchestrator.workers.errors import CharacterizationError
from orchestrator.workers.generation import Generation

# Ground truth the fit must recover.
INTERCEPT, PREFILL_A, DECODE_B = 0.05, 0.00012, 0.011


def observations(n=80, noise=0.02, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        prefill, decode = rng.randint(80, 2400), rng.randint(16, 900)
        seconds = (INTERCEPT + PREFILL_A * prefill + DECODE_B * decode
                   + rng.gauss(0, noise))
        out.append(Observation(prefill, decode, seconds))
    return out


def test_the_fit_recovers_planted_coefficients():
    fit = fit_linear(observations())
    assert fit.prefill_s_per_token == pytest.approx(PREFILL_A, rel=0.15)
    assert fit.decode_s_per_token == pytest.approx(DECODE_B, rel=0.05)
    assert fit.intercept_s == pytest.approx(INTERCEPT, abs=0.02)


def test_the_holdout_is_reported_and_is_not_the_training_set():
    """Fitting three parameters to a few dozen points can produce a high
    in-sample R-squared from noise alone."""
    fit = fit_linear(observations())
    assert fit.holdout_n > 0 and fit.holdout_r2 is not None
    assert fit.n + fit.holdout_n == 80


def test_the_fit_is_reproducible_from_the_same_observations():
    """A random holdout would make the reported R-squared differ between two
    runs over identical data."""
    obs = observations()
    assert fit_linear(obs).holdout_r2 == fit_linear(obs).holdout_r2


def test_too_few_samples_is_refused():
    """Better than coefficients that carry an R-squared and no information."""
    with pytest.raises(CharacterizationError, match="too few"):
        fit_linear(observations(n=5))


def test_a_rank_deficient_probe_set_is_refused():
    """If prefill and decode move together the two coefficients are not
    separately identifiable."""
    degenerate = [Observation(p, p * 2, 0.05 + 0.001 * p)
                  for p in range(100, 100 + 40 * 10, 10)]
    with pytest.raises(CharacterizationError, match="separately identifiable"):
        fit_linear(degenerate)


def test_gpu_seconds_excludes_the_intercept():
    """Per-request overhead is time a caller waits for but not work the GPU
    did; charging it would inflate the apparent cost of the cheap arm."""
    coefficients = CostCoefficients(models={"m": {1: fit_linear(observations())}})
    latency = coefficients.imputed_latency_s("m", 1000, 500)
    occupancy = coefficients.gpu_seconds("m", 1000, 500)
    fit = coefficients.fit_for("m")
    assert latency - occupancy == pytest.approx(fit.intercept_s)


def test_usd_is_derived_from_the_stated_rate():
    coefficients = CostCoefficients(
        models={"m": {1: fit_linear(observations())}}, usd_per_gpu_hour=3600.0,
    )
    assert coefficients.usd("m", 1000, 500) == pytest.approx(
        coefficients.gpu_seconds("m", 1000, 500)
    )


def test_an_uncharacterized_model_raises_rather_than_borrowing():
    """Falling back to another model's coefficients produces cost numbers that
    look fine and describe different weights."""
    coefficients = CostCoefficients(models={"m": {1: fit_linear(observations())}})
    with pytest.raises(KeyError, match="no cost coefficients"):
        coefficients.fit_for("other")


def test_imputing_an_unpriced_row_leaves_it_null_rather_than_raising():
    """A sweep must run before characterization exists."""
    row = Generation(run_id="r", task_id="t", arm="a", seed=0, params_hash="p",
                     model_id="unknown")
    assert CostCoefficients().impute(row).gpu_seconds is None


def test_coefficients_round_trip_through_json(tmp_path):
    original = CostCoefficients(
        models={"m": {1: fit_linear(observations()), 8: fit_linear(observations(seed=9))}},
        hardware="A100-80GB", usd_per_gpu_hour=1.10, notes="pilot",
    )
    path = original.save(tmp_path / "cost_coefficients.json")
    loaded = CostCoefficients.load(path)
    assert loaded.hardware == "A100-80GB"
    assert loaded.usd_per_gpu_hour == 1.10
    assert loaded.fit_for("m", 8).decode_s_per_token == \
        original.fit_for("m", 8).decode_s_per_token


def test_load_or_empty_tolerates_a_missing_file(tmp_path):
    assert CostCoefficients.load_or_empty(tmp_path / "nope.json").models == {}


def test_load_raises_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="characterization"):
        CostCoefficients.load(tmp_path / "nope.json")


# -- the definition-of-done check --------------------------------------------


def _row(mode, prefill, decode, wall_s, exact=True, batch=1):
    return Generation(
        run_id="r", task_id="t", arm="a", seed=0, params_hash="p",
        model_id="m", prefill_tokens=prefill, decode_tokens=decode,
        wall_ms=wall_s * 1000, mode=mode, batch_size=batch, tokens_exact=exact,
    )


def _coefficients():
    return CostCoefficients(models={"m": {1: fit_linear(observations())}})


def test_validation_passes_when_imputation_tracks_measurement():
    rows = [_row("serving", o.prefill_tokens, o.decode_tokens, o.seconds)
            for o in observations()]
    report = validate_imputation(rows, _coefficients())[0]
    assert report.passed and report.r2 > 0.9


def test_sweep_rows_are_excluded_from_validation():
    """A batched sweep's wall-clock measures queue depth; regressing against
    it produces a number that is not about latency at all."""
    rows = [_row("sweep", o.prefill_tokens, o.decode_tokens, 900.0)
            for o in observations()]
    assert validate_imputation(rows, _coefficients()) == []


def test_approximate_token_rows_are_excluded():
    """The regressor would be measuring the tokenizer's error."""
    rows = [_row("serving", o.prefill_tokens, o.decode_tokens, o.seconds, exact=False)
            for o in observations()]
    assert validate_imputation(rows, _coefficients()) == []


def test_validation_can_be_restricted_to_one_concurrency():
    rows = (
        [_row("serving", o.prefill_tokens, o.decode_tokens, o.seconds, batch=1)
         for o in observations()]
        + [_row("serving", o.prefill_tokens, o.decode_tokens, o.seconds * 3, batch=8)
           for o in observations()]
    )
    assert validate_imputation(rows, _coefficients(), batch=1)[0].n == 80


def test_a_bad_imputation_fails_loudly():
    """If R-squared is low the imputation is fiction, and that has to be said
    before R4 builds a frontier on it."""
    rows = [_row("serving", o.prefill_tokens, o.decode_tokens,
                 random.Random(o.prefill_tokens).uniform(0.1, 40.0))
            for o in observations()]
    assert not validate_imputation(rows, _coefficients())[0].passed


def test_r2_is_zero_when_there_is_no_variance_to_explain():
    """Reporting 1.0 would overstate the model."""
    coefficients = CostCoefficients(models={"m": {1: LinearFit(
        intercept_s=1.0, prefill_s_per_token=0.0, decode_s_per_token=0.0,
        n=10, r2=1.0, rmse_s=0.0,
    )}})
    rows = [_row("serving", 100, 100, 1.0) for _ in range(20)]
    assert validate_imputation(rows, coefficients)[0].r2 == 0.0
