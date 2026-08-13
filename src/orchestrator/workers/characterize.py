"""The characterization pass — measure latency once, per model, per concurrency.

Run against the **OpenAI-compatible server**, never against a sweep. A sweep is
tuned for throughput and runs hundreds of requests concurrently, so its
wall-clock measures queue depth. Here every request is timed individually at a
declared, fixed concurrency, which is the only setting where elapsed time is a
property of the model.

The probe set is a **grid over prefill length and max_tokens**, not a sample of
real tasks. Real prompts correlate length with content, and if prefill and
decode move together the two coefficients are not separately identifiable — the
fit is rank-deficient and `fit_linear` refuses it. A grid varies them
independently by construction, which is the entire reason to use synthetic
probes for this and real tasks for everything else.

Batch 1 and batch 8 are both measured. Batch 1 defines cost and imputed
latency; batch 8 exists so the gap between them is documented rather than
assumed, and so a serving deployment at concurrency 8 has a number of its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .backends.base import Backend, GenRequest
from .cost import (
    CostCoefficients, ImputationReport, LinearFit, Observation, fit_linear,
)
from .errors import CharacterizationError
from .generation import Generation
from .params import GenParams
from .tokenization import get_tokenizer

# Prefill targets, in tokens. Spread across the range a real task prompt spans:
# a bare MBPP prompt is a couple of hundred tokens, one carrying visible tests
# runs to a couple of thousand.
DEFAULT_PREFILL_TARGETS: tuple[int, ...] = (128, 384, 1024, 2048)

# Decode caps. The fit regresses on *actual* decode tokens, so a probe that
# stops early is still a valid observation — the cap only shapes the spread.
DEFAULT_MAX_TOKENS: tuple[int, ...] = (64, 256, 512, 1024)

DEFAULT_CONCURRENCIES: tuple[int, ...] = (1, 8)

# Repeats per grid cell. Server-side variance at fixed tokens is real, and one
# sample per cell fits the noise as though it were signal.
DEFAULT_REPEATS = 3

_FILLER = (
    "The function should handle empty input, duplicate elements, and negative "
    "numbers correctly, and should not mutate its arguments. "
)

_PROBE_SYSTEM = (
    "You are a Python programmer. Answer with code only, in a single block."
)


@dataclass(frozen=True)
class ProbeSpec:
    """One cell of the grid."""

    prefill_target: int
    max_tokens: int
    repeat: int = 0


@dataclass
class CharacterizationResult:
    """Everything one pass produced, fits and diagnostics together."""

    coefficients: CostCoefficients
    fits: dict[str, dict[int, LinearFit]] = field(default_factory=dict)
    validation: list[ImputationReport] = field(default_factory=list)
    n_requests: int = 0
    n_failed: int = 0
    duration_s: float = 0.0
    generations: list[Generation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every model cleared R1's definition-of-done threshold."""
        return bool(self.validation) and all(r.passed for r in self.validation)

    def report(self) -> str:
        lines = [
            f"characterization: {self.n_requests} requests "
            f"({self.n_failed} failed) in {self.duration_s:.1f}s",
        ]
        for model, by_batch in sorted(self.fits.items()):
            for batch, fit in sorted(by_batch.items()):
                holdout = (
                    f", holdout R^2={fit.holdout_r2:.4f} (n={fit.holdout_n})"
                    if fit.holdout_r2 is not None else ""
                )
                lines.append(
                    f"  {model} @ batch={batch}: "
                    f"intercept={fit.intercept_s:.4f}s "
                    f"prefill={fit.prefill_s_per_token * 1000:.4f}ms/tok "
                    f"decode={fit.decode_s_per_token * 1000:.4f}ms/tok "
                    f"| n={fit.n} R^2={fit.r2:.4f}{holdout}"
                )
        for rep in self.validation:
            lines.append(f"  {rep.summary}")
        return "\n".join(lines)


def build_probe_prompt(target_tokens: int, count: Callable[[str], int]) -> str:
    """Grow a prompt until it reaches roughly `target_tokens`.

    Counted with the **serving tokenizer**, because prefill is the regressor:
    an estimate here would put error on the x-axis of the fit, which biases the
    slope rather than merely adding noise around it.
    """
    base = (
        "Write a Python function that processes a list of integers and returns "
        "a summary. "
    )
    if count(base) >= target_tokens:
        return base

    # Overshoot generously once, then binary-search the word count. The
    # tokenizer is the slow part of building a probe set, and the linear
    # alternative costs one call per word — hundreds of calls per probe, on
    # strings thousands of characters long.
    words = (base + _FILLER * (target_tokens // 4 + 8)).split(" ")
    if count(" ".join(words)) <= target_tokens:
        return " ".join(words)

    lo, hi = 8, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(" ".join(words[:mid])) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def build_probes(prefill_targets: Sequence[int] = DEFAULT_PREFILL_TARGETS,
                 max_tokens: Sequence[int] = DEFAULT_MAX_TOKENS,
                 repeats: int = DEFAULT_REPEATS) -> list[ProbeSpec]:
    """The full grid, in a fixed order so a pass is reproducible."""
    return [
        ProbeSpec(prefill_target=p, max_tokens=m, repeat=r)
        for p in prefill_targets
        for m in max_tokens
        for r in range(repeats)
    ]


def characterize(backend: Backend, *,
                 roles: Sequence[str] = ("small", "large"),
                 concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
                 probes: Sequence[ProbeSpec] | None = None,
                 hardware: str = "unspecified",
                 usd_per_gpu_hour: float = 0.0,
                 require_exact_tokens: bool = True,
                 threshold: float = 0.9,
                 notes: str = "") -> CharacterizationResult:
    """Measure a serving backend and fit its cost coefficients.

    Refuses a `mode == "sweep"` backend outright. That check is the single most
    important line in this module: characterizing against offline batch
    generation would fit latency coefficients to queue depth, and every latency
    number in the project would be wrong in a way that looks like plausible
    data.
    """
    if backend.mode != "serving":
        raise CharacterizationError(
            f"backend {backend.name!r} runs in {backend.mode!r} mode; "
            f"characterization requires a serving backend at declared "
            f"concurrency. Wall-clock from batched offline generation measures "
            f"queue depth, not the model."
        )

    probes = list(probes if probes is not None else build_probes())
    if not probes:
        raise CharacterizationError("empty probe set")

    started = time.perf_counter()
    fits: dict[str, dict[int, LinearFit]] = {}
    all_gens: list[Generation] = []
    n_requests = 0
    n_failed = 0

    for role in roles:
        model_id = backend.model_id(role)
        # Only reach for the real tokenizer when exactness is actually
        # required. A caller who has already accepted approximate counts
        # should not pay a ten-second torch import, or risk a hub lookup, to
        # get a number they said they did not need.
        tokenizer = get_tokenizer(
            model_id,
            require_exact=require_exact_tokens,
            allow_hf=require_exact_tokens,
        )
        # One prompt per distinct prefill target, not per probe. The grid
        # repeats each target across every max_tokens value and every repeat,
        # so building per probe would do the same work a dozen times over.
        prompts = {
            target: build_probe_prompt(target, tokenizer.count)
            for target in sorted({spec.prefill_target for spec in probes})
        }

        for concurrency in concurrencies:
            requests = [
                GenRequest(
                    task_id=f"probe-p{spec.prefill_target}-m{spec.max_tokens}-r{spec.repeat}",
                    arm=f"characterize_{role}",
                    seed=spec.repeat,
                    model_role=role,
                    system=_PROBE_SYSTEM,
                    user=prompts[spec.prefill_target],
                    # Greedy, so decode length is driven by the cap rather than
                    # by sampling luck, and two repeats of a cell differ only
                    # in server-side timing.
                    params=GenParams(temperature=0.0, top_p=1.0,
                                     max_tokens=spec.max_tokens),
                )
                for spec in probes
            ]
            raws = backend.generate(requests, concurrency=concurrency)
            n_requests += len(raws)

            observations: list[Observation] = []
            for req, raw in zip(requests, raws):
                gen = Generation(
                    run_id="characterization",
                    task_id=req.task_id, arm=req.arm, seed=req.seed,
                    params_hash=req.params_hash,
                    text=raw.text, model_id=raw.model_id,
                    prefill_tokens=raw.prefill_tokens,
                    decode_tokens=raw.decode_tokens,
                    wall_ms=raw.wall_ms, finish_reason=raw.finish_reason,
                    mode=raw.mode, batch_size=raw.batch_size,
                    backend=backend.name, tokens_exact=raw.tokens_exact,
                    error=raw.error,
                )
                all_gens.append(gen)

                if raw.finish_reason == "error":
                    n_failed += 1
                    continue
                # A request that produced nothing carries no timing signal, and
                # a refusal returns almost instantly — including either would
                # drag the intercept toward zero.
                if raw.decode_tokens <= 0 or raw.wall_ms <= 0:
                    n_failed += 1
                    continue
                if require_exact_tokens and not raw.tokens_exact:
                    raise CharacterizationError(
                        f"token counts for {model_id!r} are approximate; "
                        f"cost coefficients fitted on estimated tokens are not "
                        f"trustworthy. Install the serving tokenizer, or pass "
                        f"require_exact_tokens=False and mark the result."
                    )
                observations.append(Observation(
                    prefill_tokens=raw.prefill_tokens,
                    decode_tokens=raw.decode_tokens,
                    seconds=raw.wall_ms / 1000.0,
                ))

            fits.setdefault(model_id, {})[concurrency] = fit_linear(observations)

    coefficients = CostCoefficients(
        models=fits,
        hardware=hardware,
        usd_per_gpu_hour=usd_per_gpu_hour,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )

    from .cost import validate_imputation

    return CharacterizationResult(
        coefficients=coefficients,
        fits=fits,
        # Validated at the reference concurrency only. Mixing batch-8 rows in
        # would test the batch-1 fit against contended timings and understate
        # an imputation that is working correctly.
        validation=validate_imputation(
            all_gens, coefficients, threshold=threshold,
            batch=coefficients.reference_batch,
        ),
        n_requests=n_requests,
        n_failed=n_failed,
        duration_s=time.perf_counter() - started,
        generations=all_gens,
    )


def run_and_save(backend: Backend, path: Path | str, *, merge: bool = True,
                 **kwargs: Any) -> CharacterizationResult:
    """Characterize and write `cost_coefficients.json`.

    The file is written even when validation fails, and the caller is expected
    to surface that. Withholding a failing fit hides the diagnosis: a low
    R-squared is information about the serving stack, and the coefficients are
    what someone needs in order to investigate it.

    Coefficients for models this pass did not touch are carried over rather
    than dropped. One vLLM server hosts one model, so characterizing an arm
    means pointing at a server that has only that arm's weights resident — and
    on a single card the two arms cannot even be resident at once without
    contending, which would fit the coefficients to queue depth. Both arms
    therefore land in the file one pass at a time, and overwriting would mean
    the second pass silently deleted the first.
    """
    result = characterize(backend, **kwargs)
    coefficients = result.coefficients
    if merge and Path(path).exists():
        carried = {
            model: fits
            for model, fits in CostCoefficients.load(path).models.items()
            if model not in coefficients.models
        }
        if carried:
            coefficients = replace(
                coefficients, models={**carried, **coefficients.models},
            )
    coefficients.save(path)
    return result
