"""Cost imputation onto a sealed run, without touching a single row.

This is what makes "re-characterize on new hardware and the rollouts do not
need regenerating" true rather than aspirational.
"""

from __future__ import annotations

import pytest

from orchestrator.workers.cost import CostCoefficients, LinearFit
from orchestrator.workers.impute import (
    fingerprint, impute_run, list_sidecars, read_imputed,
)
from orchestrator.workers.store import read_rows
from orchestrator.workers.sweep import run_sweep


def coefficients(decode_b=1e-2, usd_per_gpu_hour=1.10):
    fit = LinearFit(intercept_s=0.05, prefill_s_per_token=1e-4,
                    decode_s_per_token=decode_b, n=50, r2=0.99, rmse_s=0.01)
    return CostCoefficients(
        models={"mock-small": {1: fit}, "mock-large": {1: fit}},
        hardware="A100-80GB", usd_per_gpu_hour=usd_per_gpu_hour,
    )


def test_a_sidecar_covers_every_row(sweep_config):
    report = run_sweep(sweep_config)
    result = impute_run(sweep_config.out_root, report.run_id, coefficients())
    assert result.n_rows == 96 and result.n_imputed == 96
    assert result.coverage == 1.0
    assert result.total_gpu_seconds > 0 and result.total_usd > 0


def test_imputation_does_not_touch_the_rows(sweep_config):
    """A sealed run's manifest carries per-file checksums; rewriting rows to
    add two columns would invalidate every number already computed from it."""
    report = run_sweep(sweep_config)
    before = [r["rollout_id"] for r in read_rows(sweep_config.out_root, report.run_id)]
    parts = {p: p.read_bytes() for p in
             (sweep_config.out_root / report.run_id / "generations").glob("*.jsonl")}
    impute_run(sweep_config.out_root, report.run_id, coefficients())
    after = [r["rollout_id"] for r in read_rows(sweep_config.out_root, report.run_id)]
    assert before == after
    assert all(path.read_bytes() == data for path, data in parts.items())


def test_two_costings_of_one_run_coexist(sweep_config):
    """The A100 costing and the H100 costing of the same generations."""
    report = run_sweep(sweep_config)
    impute_run(sweep_config.out_root, report.run_id, coefficients(decode_b=1e-2))
    impute_run(sweep_config.out_root, report.run_id, coefficients(decode_b=2e-2))
    assert len(list_sidecars(sweep_config.out_root, report.run_id)) == 2


def test_the_fingerprint_follows_the_fitted_numbers():
    assert fingerprint(coefficients(decode_b=1e-2)) != \
        fingerprint(coefficients(decode_b=2e-2))


def test_the_fingerprint_follows_the_dollar_rate():
    assert fingerprint(coefficients(usd_per_gpu_hour=1.1)) != \
        fingerprint(coefficients(usd_per_gpu_hour=2.2))


def test_the_fingerprint_ignores_timestamps_and_notes():
    """Re-running an identical characterization should reuse the sidecar
    rather than producing a second copy under a new name."""
    a, b = coefficients(), coefficients()
    a.created_at, a.notes = "2026-01-01T00:00:00+00:00", "first"
    b.created_at, b.notes = "2026-06-01T00:00:00+00:00", "second"
    assert fingerprint(a) == fingerprint(b)


def test_an_existing_sidecar_is_not_rewritten(sweep_config):
    report = run_sweep(sweep_config)
    first = impute_run(sweep_config.out_root, report.run_id, coefficients())
    mtime = first.path.stat().st_mtime_ns
    second = impute_run(sweep_config.out_root, report.run_id, coefficients())
    assert second.path.stat().st_mtime_ns == mtime
    assert second.n_imputed == first.n_imputed


def test_unpriced_models_are_reported_not_silently_skipped(sweep_config):
    """Cost numbers must not be reported for models nobody characterized."""
    report = run_sweep(sweep_config)
    partial = coefficients()
    del partial.models["mock-large"]
    result = impute_run(sweep_config.out_root, report.run_id, partial)
    assert result.n_unpriced == 48
    assert result.unpriced_models == {"mock-large"}
    assert "unpriced" in result.summary()


def test_sidecar_entries_carry_identity(sweep_config):
    report = run_sweep(sweep_config)
    result = impute_run(sweep_config.out_root, report.run_id, coefficients())
    entry = next(iter(read_imputed(result.path)))
    for field in ("rollout_id", "task_id", "arm", "seed", "model_id",
                  "gpu_seconds", "imputed_latency_s", "usd"):
        assert field in entry


def test_imputed_latency_exceeds_gpu_seconds_by_the_intercept(sweep_config):
    report = run_sweep(sweep_config)
    result = impute_run(sweep_config.out_root, report.run_id, coefficients())
    for entry in read_imputed(result.path):
        assert entry["imputed_latency_s"] - entry["gpu_seconds"] == pytest.approx(0.05)


def test_no_sidecars_before_imputation(sweep_config):
    report = run_sweep(sweep_config)
    assert list_sidecars(sweep_config.out_root, report.run_id) == []
