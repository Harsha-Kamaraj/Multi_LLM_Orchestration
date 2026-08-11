"""The sweep runner — one command, resumable, append-only.

Three properties, all required, all enforced here rather than documented:

**One command.** If running a sweep needs a checklist, it will be run
inconsistently, and two runs that differ in a step nobody wrote down are two
experiments wearing one `run_id`.

**Resumable.** Sweeps take hours and will be interrupted. Resume is keyed on
`(task_id, arm, seed, params_hash)`: a completed cell is never recomputed, and
a cell computed under different parameters is never mistaken for a hit.

**Append-only.** Rows are appended, partitioned by `run_id`, and never mutated.
A re-run under changed conditions is a new `run_id`, full stop — which is
automatic here, because every arm's `params_hash` is folded into the config
that the `run_id` hashes.

Cells are generated in a deterministic order and grouped by arm, so an offline
vLLM engine serves one model per group. Interleaving arms would swap weights
between batches, which is how a sweep ends up with two models resident and an
out-of-memory error four hours in.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..types import Task
from .arms import Arm, DEFAULT_SEEDS, FROZEN_LADDER, resolve_arms
from .backends import get_backend
from .backends.base import Backend, GenRequest
from .corpus import CorpusView, build_corpus
from .cost import CostCoefficients, DEFAULT_COEFFICIENTS_PATH
from .errors import WorkerError
from .extract import extract
from .generation import Generation
from .prompts import PromptContext
from .resume import CellKey, ResumeIndex, build_resume_index, summarize
from .runid import make_run_id
from .store import RolloutStore

log = logging.getLogger("orchestrator.workers.sweep")

DEFAULT_OUT_ROOT = Path("runs")

# Truncation above this fraction stops the sweep. `finish_reason == "length"`
# grades as a failure and looks exactly like a capability gap, so a run that is
# quietly truncating 20% of its generations produces a believable, wrong
# accuracy number. Better to fail loudly and raise max_tokens.
DEFAULT_TRUNCATION_ALARM = 0.15


@dataclass
class SweepConfig:
    """Everything that shapes a sweep. Hashed into the `run_id`."""

    tasks_path: Path | str
    splits_path: Path | str | None = None
    out_root: Path | str = DEFAULT_OUT_ROOT

    arms: tuple[str, ...] = FROZEN_LADDER
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    include_splits: tuple[str, ...] | None = None
    limit: int | None = None

    backend: str | None = None
    small_model: str = "mock-small"
    large_model: str = "mock-large"
    backend_options: dict[str, Any] = field(default_factory=dict)

    #: Requests handed to the backend at once. For offline vLLM this is the
    #: batch; for a serving backend it is the in-flight concurrency.
    batch_size: int = 32
    coefficients_path: Path | str | None = DEFAULT_COEFFICIENTS_PATH
    resume: bool = True
    retry_errors: bool = True
    truncation_alarm: float = DEFAULT_TRUNCATION_ALARM
    dataset: str = ""

    def identity(self, corpus_fingerprint: str = "") -> dict[str, Any]:
        """The subset of config that defines the experiment.

        Deliberately excludes output paths, batch size, and resume behaviour:
        those change how a sweep runs, not what it produces, and folding them
        in would make an identical experiment resumed with a different batch
        size land in a different run directory.

        Two things *are* included, and both close a hole:

        * **`params_hash` per arm**, so a template or sampling change produces
          a new `run_id` automatically.
        * **the corpus fingerprint**, so regenerating the task manifest does
          too. Hashing only the path would let R2 rewrite the manifest and
          have the new tasks land in a run whose sealed sibling describes the
          previous corpus.
        """
        arms = resolve_arms(list(self.arms))
        return {
            "arms": {a.name: a.params_hash for a in arms},
            "seeds": list(self.seeds),
            "tasks_path": str(self.tasks_path),
            "corpus_fingerprint": corpus_fingerprint,
            "splits_path": str(self.splits_path) if self.splits_path else None,
            "include_splits": list(self.include_splits) if self.include_splits else None,
            "limit": self.limit,
            "backend": self.backend,
            "small_model": self.small_model,
            "large_model": self.large_model,
            "dataset": self.dataset,
        }


@dataclass
class SweepReport:
    """What a sweep did, for the log line and the manifest."""

    run_id: str
    planned: int
    generated: int
    skipped: int
    failed: int
    truncated: int
    duration_s: float
    manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def truncation_rate(self) -> float:
        return self.truncated / self.generated if self.generated else 0.0

    def summary(self) -> str:
        rate = f"{self.truncation_rate:.2%}"
        return (
            f"run {self.run_id}: generated {self.generated}, "
            f"skipped {self.skipped} already done, {self.failed} failed, "
            f"{self.truncated} truncated ({rate}) in {self.duration_s:.1f}s"
        )


def plan_cells(corpus: CorpusView, arms: Sequence[Arm],
               seeds: Sequence[int]) -> list[tuple[Task, Arm, int]]:
    """Build the work list in a fixed order: arm, then task, then seed.

    Arm-major so an offline engine loads each model exactly once. The order is
    deterministic, so an interrupted sweep resumes where it stopped rather than
    scattering across the corpus — which matters when a sweep is killed early
    and the partial results are used for a smoke check.
    """
    return [
        (task, arm, seed)
        for arm in arms
        for task in corpus.tasks
        for seed in seeds
    ]


def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _preflight(config: SweepConfig, arms: Sequence[Arm],
               backend: Backend) -> list[str]:
    """Checks that are cheap now and expensive to discover mid-sweep."""
    warnings: list[str] = []

    repair = [a.name for a in arms if a.requires_parent]
    if repair:
        raise WorkerError(
            f"arms {repair} sit above ladder step 0 and need a parent "
            f"generation plus its visible-test feedback. Test outcomes come "
            f"from R2's grader, so a repair ladder cannot run inside the sweep "
            f"that produced its parents — it is a second sweep over a graded "
            f"run. That is Phase 2 work and is not wired up yet."
        )

    greedy = [a.name for a in arms if a.params.temperature == 0.0]
    if len(config.seeds) > 1 and greedy and not backend.honors_seed:
        warnings.append(
            f"backend {backend.name!r} ignores seeds, and arms {greedy} run at "
            f"temperature 0, so all {len(config.seeds)} seeds will return "
            f"identical text. You are paying {len(config.seeds)}x for one "
            f"distinct generation, and R4's cluster bootstrap will see no "
            f"within-task variance. Use one seed, or raise temperature."
        )
    return warnings


def run_sweep(config: SweepConfig, *, backend: Backend | None = None,
              progress_every: int = 10) -> SweepReport:
    """Generate every planned cell that is not already done, then seal the run.

    A backend may be injected; otherwise one is constructed from the config.
    An injected backend is not closed here, because the caller that built it
    owns its lifetime — and for vLLM that lifetime is tens of gigabytes of VRAM.
    """
    started = time.perf_counter()
    corpus = build_corpus(
        config.tasks_path, config.splits_path,
        include_splits=config.include_splits, limit=config.limit,
    )
    arms = resolve_arms(list(config.arms))

    owns_backend = backend is None
    if backend is None:
        backend = get_backend(
            config.backend,
            small_model=config.small_model,
            large_model=config.large_model,
            **config.backend_options,
        )

    try:
        warnings = _preflight(config, arms, backend)
        for warning in warnings:
            log.warning("%s", warning)

        identity = config.identity(corpus.fingerprint)
        run_id = make_run_id(identity)
        planned_cells = plan_cells(corpus, arms, config.seeds)

        index = _resume_index(config, run_id, arms)
        log.info("sweep %s: %s", run_id, summarize(index, len(planned_cells)))
        log.info("corpus: %d tasks, splits %s", len(corpus.tasks), corpus.counts())

        coefficients = _load_coefficients(config)
        _teach_mock(backend, corpus)

        store = RolloutStore(config.out_root, run_id)
        if store.is_sealed:
            # Re-running an identical sweep is a no-op, not an error. The
            # config hashes into the run_id, so an already-sealed run under
            # this id holds exactly the experiment being asked for.
            done = sum(1 for cell in planned_cells if index.has(_key(*cell)))
            if done >= len(planned_cells):
                log.info("run %s is already complete and sealed", run_id)
                return SweepReport(
                    run_id=run_id, planned=len(planned_cells), generated=0,
                    skipped=len(planned_cells), failed=0, truncated=0,
                    duration_s=time.perf_counter() - started,
                    manifest=_read_manifest(config.out_root, run_id),
                    warnings=warnings,
                )
            raise WorkerError(
                f"run {run_id} is sealed but only {done}/{len(planned_cells)} "
                f"planned cells are present. A sealed run is immutable, so it "
                f"cannot be extended — the manifest's counts and checksums "
                f"already describe it. Widen the sweep under a new run_id."
            )
        store.open(config={"identity": identity, "sweep": _serializable(config)})

        generated = skipped = failed = truncated = 0
        try:
            for arm in arms:
                cells = [
                    (task, a, seed) for task, a, seed in planned_cells
                    if a.name == arm.name
                ]
                todo = [
                    (task, a, seed) for task, a, seed in cells
                    if not index.has(_key(task, a, seed))
                ]
                skipped += len(cells) - len(todo)
                if not todo:
                    log.info("arm %s: nothing to do", arm.name)
                    continue

                log.info("arm %s: %d cells to generate", arm.name, len(todo))
                arm_done = arm_truncated = 0
                for n, chunk in enumerate(_chunks(todo, config.batch_size), start=1):
                    gens = _generate_chunk(
                        chunk, backend, corpus, run_id, config, coefficients,
                    )
                    for gen in gens:
                        store.append(gen)
                        generated += 1
                        arm_done += 1
                        failed += int(gen.finish_reason == "error")
                        truncated += int(gen.truncated)
                        arm_truncated += int(gen.truncated)
                    # Flushed per chunk, not per row: a chunk is the unit of
                    # work a resume would repeat anyway, so syncing more often
                    # buys nothing and costs throughput.
                    store.flush()
                    if n % progress_every == 0 or n == 1:
                        log.info(
                            "  %s: %d/%d cells, %.1f%% truncated",
                            arm.name, arm_done, len(todo),
                            100.0 * arm_truncated / max(1, arm_done),
                        )
        finally:
            store.close()

        rate = truncated / generated if generated else 0.0
        if rate > config.truncation_alarm:
            warnings.append(
                f"truncation rate {rate:.2%} exceeds the {config.truncation_alarm:.0%} "
                f"alarm. Truncated generations grade as failures and are "
                f"indistinguishable from a capability gap in the aggregate. "
                f"Raise max_tokens and re-run before anyone reads an accuracy "
                f"number off this run."
            )
            log.error("%s", warnings[-1])

        manifest = store.seal(
            config={"identity": identity, "sweep": _serializable(config)},
            extra={
                "warnings": warnings,
                "corpus_counts": corpus.counts(),
                "resume": summarize(index, len(planned_cells)),
                "backend": backend.name,
                "backend_mode": backend.mode,
                "honors_seed": backend.honors_seed,
            },
        )

        report = SweepReport(
            run_id=run_id, planned=len(planned_cells), generated=generated,
            skipped=skipped, failed=failed, truncated=truncated,
            duration_s=time.perf_counter() - started,
            manifest=manifest, warnings=warnings,
        )
        log.info("%s", report.summary())
        return report
    finally:
        if owns_backend:
            backend.close()


# -- internals ---------------------------------------------------------------


def _key(task: Task, arm: Arm, seed: int) -> CellKey:
    return (task.task_id, arm.name, seed, arm.params_hash)


def _resume_index(config: SweepConfig, run_id: str,
                  arms: Sequence[Arm]) -> ResumeIndex:
    if not config.resume:
        return ResumeIndex(run_id=run_id)
    index = build_resume_index(
        config.out_root, run_id, retry_errors=config.retry_errors,
    )
    # Should never fire: each arm's params_hash is in the run config, so a
    # parameter change lands in a different run_id. If it does fire, the store
    # holds two experiments and must not be extended.
    index.check_params({a.name: a.params_hash for a in arms})
    return index


def _load_coefficients(config: SweepConfig) -> CostCoefficients:
    """Load cost coefficients if they exist; impute nothing if they do not.

    A sweep must run before characterization has happened — that is the point
    of imputing from stored token counts rather than measuring during the
    sweep. The fields stay null and a later pass fills them in.
    """
    if config.coefficients_path is None:
        return CostCoefficients()
    coefficients = CostCoefficients.load_or_empty(config.coefficients_path)
    if not coefficients.models:
        log.info(
            "no cost coefficients at %s; gpu_seconds and imputed_latency_s will "
            "be null and can be filled in after characterization",
            config.coefficients_path,
        )
    return coefficients


def _teach_mock(backend: Backend, corpus: CorpusView) -> None:
    """Give the mock backend the corpus's reference solutions, if it wants them."""
    register = getattr(backend, "register_solution", None)
    if register is None:
        return
    for task in corpus.tasks:
        solution = task.metadata.get("mock_solution")
        if solution:
            register(task.task_id, str(solution))


def _generate_chunk(chunk: Sequence[tuple[Task, Arm, int]], backend: Backend,
                    corpus: CorpusView, run_id: str, config: SweepConfig,
                    coefficients: CostCoefficients) -> list[Generation]:
    requests = []
    for task, arm, seed in chunk:
        system, user = arm.params.template.render(PromptContext.from_task(task))
        requests.append(GenRequest(
            task_id=task.task_id, arm=arm.name, seed=seed,
            model_role=arm.model_role, system=system, user=user,
            params=arm.params, ladder_step=arm.ladder_step,
        ))

    raws = backend.generate(requests, concurrency=config.batch_size)

    out: list[Generation] = []
    for (task, arm, seed), raw in zip(chunk, raws):
        # Extraction happens here, on R1's side of the fence. R2 receives
        # `code` and never parses raw model output, so a prompt-format change
        # cannot become a bug in someone else's file.
        extraction = extract(raw.text, entrypoint=task.entrypoint)
        gen = Generation(
            run_id=run_id,
            task_id=task.task_id,
            arm=arm.name,
            seed=seed,
            params_hash=arm.params_hash,
            text=raw.text,
            code=extraction.code,
            model_id=raw.model_id,
            prefill_tokens=raw.prefill_tokens,
            decode_tokens=raw.decode_tokens,
            wall_ms=raw.wall_ms,
            finish_reason=raw.finish_reason,
            mode=raw.mode,
            batch_size=raw.batch_size,
            backend=backend.name,
            tokens_exact=raw.tokens_exact,
            extract_strategy=extraction.strategy,
            code_parses=extraction.parses,
            split=corpus.split_of(task.task_id),
            dataset=config.dataset or str(task.metadata.get("dataset", "")),
            ladder_step=arm.ladder_step,
            created_at=_now(),
            error=raw.error,
            extra=dict(raw.extra),
        )
        out.append(coefficients.impute(gen))
    return out


def _read_manifest(root: Path | str, run_id: str) -> dict[str, Any]:
    from .store import read_manifest

    try:
        return read_manifest(root, run_id)
    except WorkerError:
        return {}


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _serializable(config: SweepConfig) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(config).items()}
