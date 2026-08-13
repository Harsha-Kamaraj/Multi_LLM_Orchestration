"""The characterization pass, and the refusal that keeps latency honest."""

from __future__ import annotations

import pytest

from orchestrator.workers.backends import get_backend
from orchestrator.workers.characterize import (
    build_probe_prompt, build_probes, characterize, run_and_save,
)
from orchestrator.workers.errors import CharacterizationError
from orchestrator.workers.tokenization import ApproxTokenizer


def serving_backend(**kwargs):
    return get_backend("mock", mode="serving", **kwargs)


def test_a_sweep_backend_is_refused():
    """The single most important line in the module: characterizing against
    offline batch generation fits latency coefficients to queue depth, and
    every latency number in the project becomes wrong in a way that looks
    like plausible data."""
    with pytest.raises(CharacterizationError, match="serving backend"):
        characterize(get_backend("mock"))


def test_the_probe_grid_varies_prefill_and_decode_independently():
    """Correlated regressors make the two coefficients unidentifiable."""
    probes = build_probes()
    prefills = {p.prefill_target for p in probes}
    for prefill in prefills:
        caps = {p.max_tokens for p in probes if p.prefill_target == prefill}
        assert len(caps) > 1


def test_the_probe_grid_is_deterministic():
    assert build_probes() == build_probes()


@pytest.mark.parametrize("target", [128, 384, 1024, 2048])
def test_probe_prompts_hit_their_target_length(target):
    prompt = build_probe_prompt(target, ApproxTokenizer().count)
    assert ApproxTokenizer().count(prompt) == pytest.approx(target, rel=0.05)


def test_probe_building_is_logarithmic_in_length():
    """A linear trim costs one tokenizer call per word, on strings thousands
    of characters long."""
    calls = []
    tokenizer = ApproxTokenizer()

    def counting(text):
        calls.append(1)
        return tokenizer.count(text)

    build_probe_prompt(2048, counting)
    assert len(calls) < 30


def test_a_pass_fits_every_model_at_every_concurrency():
    result = characterize(serving_backend(), concurrencies=(1, 8),
                          require_exact_tokens=False)
    assert set(result.fits) == {"mock-small", "mock-large"}
    for by_batch in result.fits.values():
        assert set(by_batch) == {1, 8}


def test_fits_report_a_holdout():
    result = characterize(serving_backend(), concurrencies=(1,),
                          require_exact_tokens=False)
    fit = result.fits["mock-small"][1]
    assert fit.holdout_r2 is not None and fit.holdout_n > 0


def test_fitted_coefficients_recover_the_mock_timing_model():
    """The mock's wall-clock is 40ms + 0.08ms/prefill + 6ms/decode."""
    result = characterize(serving_backend(), concurrencies=(1,),
                          require_exact_tokens=False)
    fit = result.fits["mock-small"][1]
    assert fit.decode_s_per_token == pytest.approx(0.006, rel=0.1)
    assert fit.prefill_s_per_token == pytest.approx(0.00008, rel=0.5)


def test_validation_runs_at_the_reference_concurrency_only():
    """Mixing batch-8 rows in would test the batch-1 fit against contended
    timings and understate an imputation that is working."""
    result = characterize(serving_backend(), concurrencies=(1, 8),
                          require_exact_tokens=False)
    assert result.validation
    for report in result.validation:
        assert report.threshold == 0.9


def test_approximate_tokens_are_refused_when_exactness_is_required():
    """Cost coefficients fitted on estimated tokens are not trustworthy."""
    with pytest.raises(RuntimeError, match="exact"):
        characterize(serving_backend(), concurrencies=(1,),
                     require_exact_tokens=True)


def test_failed_and_empty_generations_are_excluded_from_the_fit():
    """A refusal returns almost instantly and would drag the intercept to
    zero; a request that produced nothing carries no timing signal."""
    result = characterize(
        serving_backend(refusal_rate=0.08, error_rate=0.04),
        concurrencies=(1,), probes=build_probes(repeats=6),
        require_exact_tokens=False,
    )
    assert result.n_failed > 0
    assert result.fits["mock-small"][1].n < result.n_requests


def test_dropping_too_many_probes_fails_rather_than_fitting_noise():
    """When failures gut the grid the design goes rank-deficient, and the fit
    must refuse instead of emitting coefficients that look plausible."""
    with pytest.raises(CharacterizationError):
        characterize(
            serving_backend(refusal_rate=0.5, error_rate=0.3),
            concurrencies=(1,), require_exact_tokens=False,
        )


def test_coefficients_are_written_even_when_validation_fails(tmp_path):
    """A low R-squared is information about the serving stack; withholding
    the coefficients hides the diagnosis."""
    path = tmp_path / "cost_coefficients.json"
    result = run_and_save(
        serving_backend(), path, concurrencies=(1,),
        require_exact_tokens=False, threshold=0.999,
        hardware="mock-host", usd_per_gpu_hour=1.10,
    )
    assert not result.passed
    assert path.exists()


def test_saved_coefficients_record_the_hardware_and_rate(tmp_path):
    """USD is an assumption, recorded so a reader can substitute their own."""
    from orchestrator.workers.cost import CostCoefficients

    path = tmp_path / "cost_coefficients.json"
    run_and_save(serving_backend(), path, concurrencies=(1,),
                 require_exact_tokens=False, hardware="A100-80GB",
                 usd_per_gpu_hour=1.10, notes="pilot")
    loaded = CostCoefficients.load(path)
    assert loaded.hardware == "A100-80GB"
    assert loaded.usd_per_gpu_hour == 1.10
    assert loaded.notes == "pilot"


def test_an_empty_probe_set_is_refused():
    with pytest.raises(CharacterizationError, match="empty probe set"):
        characterize(serving_backend(), probes=[], require_exact_tokens=False)


def test_the_report_is_human_readable():
    result = characterize(serving_backend(), concurrencies=(1,),
                          require_exact_tokens=False)
    text = result.report()
    assert "mock-small" in text and "R^2" in text and "ms/tok" in text


def test_characterizing_one_role_carries_the_other_over(tmp_path):
    """One vLLM server hosts one model, and on a single card the two arms
    cannot both be resident without contending — which would fit coefficients
    to queue depth. Each arm is measured in its own pass, so a pass must not
    delete the arm it did not measure."""
    from orchestrator.workers.cost import CostCoefficients

    out = tmp_path / "cost_coefficients.json"
    probes = build_probes(repeats=6)
    backend = serving_backend()

    run_and_save(backend, out, roles=("small",), probes=probes,
                 concurrencies=(1,), require_exact_tokens=False)
    assert set(CostCoefficients.load(out).models) == {"mock-small"}

    run_and_save(backend, out, roles=("large",), probes=probes,
                 concurrencies=(1,), require_exact_tokens=False)
    assert set(CostCoefficients.load(out).models) == {"mock-small", "mock-large"}
