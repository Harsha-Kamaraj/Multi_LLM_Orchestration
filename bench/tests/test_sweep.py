"""The sweep runner's three required properties, end to end.

One command, resumable, append-only. These are integration tests on purpose:
each property is a claim about how the parts compose, and testing the parts
separately would not have caught any of the bugs these did.
"""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.workers.backends.mock import MockBackend
from orchestrator.workers.errors import WorkerError
from orchestrator.workers.store import read_generations
from orchestrator.workers.sweep import SweepConfig, plan_cells, run_sweep


def evolve(config: SweepConfig, **changes) -> SweepConfig:
    return dataclasses.replace(config, **changes)


def test_a_sweep_writes_every_planned_cell(sweep_config):
    report = run_sweep(sweep_config)
    rows = list(read_generations(sweep_config.out_root, report.run_id))
    # 24 tasks x 2 arms x 2 seeds
    assert len(rows) == 96 == report.generated == report.planned


def test_rows_carry_identity_split_and_extraction(sweep_config):
    report = run_sweep(sweep_config)
    row = next(iter(read_generations(sweep_config.out_root, report.run_id)))
    assert row.run_id == report.run_id
    assert row.dataset == "mbpp+"
    assert row.split in ("train", "val", "test")
    assert row.backend == "mock"
    assert row.extract_strategy
    assert row.model_id


def test_r1_hands_over_code_not_raw_output(sweep_config):
    """R2 grades what it is given; a prompt-format change must not become a
    bug in someone else's file."""
    report = run_sweep(sweep_config)
    rows = [r for r in read_generations(sweep_config.out_root, report.run_id)
            if r.finish_reason == "stop"]
    assert rows
    assert all(r.code and "```" not in r.code for r in rows)


def test_cells_are_planned_arm_major(sweep_config):
    """Arm-major so an offline engine loads each model exactly once;
    interleaving would swap weights between batches."""
    from orchestrator.workers.arms import resolve_arms
    from orchestrator.workers.corpus import build_corpus

    corpus = build_corpus(sweep_config.tasks_path)
    arms = resolve_arms(list(sweep_config.arms))
    names = [arm.name for _task, arm, _seed in plan_cells(corpus, arms, (0, 1))]
    assert names == ["direct_small"] * 48 + ["direct_large"] * 48


# -- resumability ------------------------------------------------------------


class Preempted(MockBackend):
    """Fails after a fixed number of backend calls, like a spot instance."""

    def __init__(self, after=2, **kwargs):
        super().__init__(**kwargs)
        self.after = after
        self.calls = 0

    def _generate(self, requests, batch_size):
        self.calls += 1
        if self.calls > self.after:
            raise KeyboardInterrupt("simulated preemption")
        return list(super()._generate(requests, batch_size))


def test_an_interrupted_sweep_leaves_an_unsealed_run(sweep_config):
    """Readers skip unsealed runs, so a partial sweep cannot be mistaken for a
    complete one."""
    from orchestrator.workers.store import RolloutStore, read_rows

    with pytest.raises(KeyboardInterrupt):
        run_sweep(sweep_config, backend=Preempted(after=2))
    run_id = next(sweep_config.out_root.iterdir()).name
    assert not RolloutStore(sweep_config.out_root, run_id).is_sealed
    assert len(list(read_rows(sweep_config.out_root, run_id, allow_unsealed=True))) == 16


def test_a_run_with_errored_cells_is_left_open_for_retry(sweep_config):
    """"Errors are retried on resume" is only reachable if the run stays open.
    Sealing one that holds retryable cells strands them: the config hashes into
    the run_id, so the identical re-run meant to retry them finds the same id
    sealed and incomplete, and refuses."""
    from orchestrator.workers.store import RolloutStore

    config = evolve(sweep_config, backend_options={"error_rate": 0.5})
    report = run_sweep(config)
    assert report.failed > 0
    assert not RolloutStore(config.out_root, report.run_id).is_sealed
    assert any("retryable" in w for w in report.warnings)

    # The identical command is what retries them, so it must run rather than
    # hard-fail on a sealed id.
    again = run_sweep(config)
    assert again.run_id == report.run_id


def test_a_clean_run_still_seals(sweep_config):
    """The open-on-error path must not stop a complete run from sealing."""
    from orchestrator.workers.store import RolloutStore

    config = evolve(sweep_config, backend_options={"error_rate": 0.0})
    report = run_sweep(config)
    assert report.failed == 0
    assert RolloutStore(config.out_root, report.run_id).is_sealed
    assert report.manifest


def test_resume_generates_only_what_is_missing(sweep_config):
    with pytest.raises(KeyboardInterrupt):
        run_sweep(sweep_config, backend=Preempted(after=2))
    report = run_sweep(sweep_config)
    assert report.skipped == 16
    assert report.generated == 80
    assert report.generated + report.skipped == 96


def test_resume_never_duplicates_a_cell(sweep_config):
    with pytest.raises(KeyboardInterrupt):
        run_sweep(sweep_config, backend=Preempted(after=2))
    report = run_sweep(sweep_config)
    keys = [r.cell_key for r in read_generations(sweep_config.out_root, report.run_id)]
    assert len(keys) == len(set(keys)) == 96


def test_resume_keeps_the_same_run_id(sweep_config):
    with pytest.raises(KeyboardInterrupt):
        run_sweep(sweep_config, backend=Preempted(after=2))
    interrupted = next(sweep_config.out_root.iterdir()).name
    assert run_sweep(sweep_config).run_id == interrupted


def test_rerunning_a_completed_sweep_is_a_no_op(sweep_config):
    """A sealed run under this id holds exactly the experiment being asked
    for, so re-running is nothing to do rather than an error."""
    first = run_sweep(sweep_config)
    second = run_sweep(sweep_config)
    assert second.run_id == first.run_id
    assert second.generated == 0 and second.skipped == 96


def test_regenerating_the_task_manifest_produces_a_new_run(sweep_config):
    """The corpus fingerprint is in the run_id, so rewriting the manifest in
    place cannot extend a run whose manifest describes the previous corpus."""
    from conftest import write_corpus

    first = run_sweep(evolve(sweep_config, limit=8))
    write_corpus(sweep_config.tasks_path, n=24, splits=("train",))
    second = run_sweep(evolve(sweep_config, limit=8))
    assert second.run_id != first.run_id


def test_widening_a_sealed_run_is_refused(sweep_config, monkeypatch):
    """A sealed run is immutable: its manifest's counts and checksums already
    describe it, so it cannot be extended even if the plan grows.

    Reaching this branch takes a monkeypatch, because everything that normally
    changes the plan is folded into the run_id and would land in a fresh
    directory. It stays as a guard against a future field being added to the
    plan without being added to `identity()`.
    """
    sealed = run_sweep(evolve(sweep_config, limit=8)).run_id

    # Force a wider plan onto the sealed run's id, which is what a future
    # plan-affecting field left out of `identity()` would do by accident.
    monkeypatch.setattr(
        "orchestrator.workers.sweep.make_run_id", lambda *a, **k: sealed,
    )
    with pytest.raises(WorkerError, match="sealed"):
        run_sweep(evolve(sweep_config, limit=8, seeds=(0, 1, 2)))


# -- append-only and run isolation -------------------------------------------


def test_changing_a_sampling_parameter_produces_a_new_run(sweep_config):
    """Each arm's params_hash is folded into the run config, so a template or
    sampling change lands in a different directory automatically."""
    first = run_sweep(sweep_config)
    second = run_sweep(evolve(sweep_config, arms=("direct_small", "probe_small")))
    assert second.run_id != first.run_id
    assert len(list(read_generations(sweep_config.out_root, first.run_id))) == 96


def test_batch_size_does_not_change_the_run_id(sweep_config):
    """Batch size changes how a sweep runs, not what it produces; folding it
    in would scatter one experiment across two directories."""
    first = run_sweep(sweep_config)
    assert run_sweep(evolve(sweep_config, batch_size=3)).run_id == first.run_id


def test_the_manifest_records_the_backend_and_seed_handling(sweep_config):
    report = run_sweep(sweep_config)
    extra = report.manifest["extra"]
    assert extra["backend"] == "mock"
    assert extra["backend_mode"] == "sweep"
    assert extra["honors_seed"] is True


# -- preflight guards --------------------------------------------------------


def test_a_seed_blind_backend_with_greedy_arms_warns(sweep_config):
    """At temperature 0 a seed-blind backend returns identical text for every
    seed: three seeds pay three times for one distinct generation."""
    class SeedBlind(MockBackend):
        honors_seed = False

    report = run_sweep(evolve(sweep_config, seeds=(0, 1, 2)), backend=SeedBlind())
    assert any("ignores seeds" in w for w in report.warnings)


def test_one_seed_does_not_warn(sweep_config):
    class SeedBlind(MockBackend):
        honors_seed = False

    report = run_sweep(evolve(sweep_config, seeds=(0,)), backend=SeedBlind())
    assert not any("ignores seeds" in w for w in report.warnings)


def test_a_repair_arm_is_refused_with_an_explanation(sweep_config):
    """Repair needs its parent's visible-test feedback, which comes from R2's
    grader — so it cannot run inside the sweep that produced its parents."""
    with pytest.raises(WorkerError, match="ladder step 0"):
        run_sweep(evolve(sweep_config, arms=("repair_small",)))


def test_a_high_truncation_rate_raises_the_alarm(sweep_config):
    """Truncated generations grade as failures and are indistinguishable from
    a capability gap in the aggregate."""
    report = run_sweep(
        evolve(sweep_config, truncation_alarm=0.01),
        backend=MockBackend(truncation_rate=0.5),
    )
    assert report.truncation_rate > 0.01
    assert any("truncation rate" in w for w in report.warnings)
    assert any("truncation" in w for w in report.manifest["extra"]["warnings"])


def test_imputation_fills_cost_when_coefficients_exist(sweep_config, tmp_path):
    from orchestrator.workers.cost import CostCoefficients, LinearFit

    fit = LinearFit(intercept_s=0.05, prefill_s_per_token=1e-4,
                    decode_s_per_token=1e-2, n=50, r2=0.99, rmse_s=0.01)
    path = CostCoefficients(
        models={"mock-small": {1: fit}, "mock-large": {1: fit}},
        usd_per_gpu_hour=1.10,
    ).save(tmp_path / "cost_coefficients.json")

    report = run_sweep(evolve(sweep_config, coefficients_path=path))
    rows = list(read_generations(sweep_config.out_root, report.run_id))
    assert all(r.gpu_seconds is not None for r in rows)
    assert all(r.imputed_latency_s > r.gpu_seconds for r in rows)


def test_a_sweep_runs_without_coefficients(sweep_config):
    """Imputing from stored token counts later is the whole point."""
    report = run_sweep(sweep_config)
    rows = list(read_generations(sweep_config.out_root, report.run_id))
    assert all(r.gpu_seconds is None for r in rows)
