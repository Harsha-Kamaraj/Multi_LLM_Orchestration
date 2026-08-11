"""The backend contract every generation engine satisfies.

A backend takes rendered prompts and returns text plus token counts. It knows
nothing about tasks, arms, seeds, resume, or the store — the sweep runner owns
all of that. Keeping the seam here is what makes "the GPU landed" a
configuration change rather than a rewrite.

Backends resolve a **role** (`small` / `large`) to a concrete `model_id`. Arms
name roles, so swapping which weights back the small rung is a backend
argument, and every row still records the model that actually ran.

Two attributes carry the sweep-versus-serving distinction outward:

* **`mode`** — `sweep` when requests are batched for throughput, `serving` when
  they run at a declared, fixed concurrency.
* **`batch_size`** on each result — how many requests were in flight.

Offline batch generation reports `mode="sweep"`, and its wall-clock is a queue
depth measurement rather than a model measurement. The characterization pass
refuses to fit on anything that is not `serving`, which is what keeps latency
coefficients honest.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..errors import BackendError
from ..generation import normalize_finish_reason
from ..params import GenParams


@dataclass(frozen=True)
class GenRequest:
    """One rendered prompt, plus the identity it will be recorded under.

    The identity fields travel with the request so a backend that reorders or
    batches internally cannot mismatch a response to a task — the association
    is carried, never inferred from position.
    """

    task_id: str
    arm: str
    seed: int
    model_role: str
    system: str
    user: str
    params: GenParams
    ladder_step: int = 0
    parent_rollout_id: str | None = None

    @property
    def params_hash(self) -> str:
        return self.params.params_hash


@dataclass(frozen=True)
class RawGeneration:
    """What a backend returns, before it becomes a `Generation`.

    Token counts come from the serving layer wherever one reports them.
    `tokens_exact=False` marks a count that came from an approximation, and
    excludes the row from cost fitting.
    """

    text: str
    model_id: str
    prefill_tokens: int = 0
    decode_tokens: int = 0
    wall_ms: float = 0.0
    finish_reason: str = "unknown"
    mode: str = "sweep"
    batch_size: int = 1
    tokens_exact: bool = True
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def failure(model_id: str, error: str, *, mode: str = "sweep",
                batch_size: int = 1, wall_ms: float = 0.0) -> "RawGeneration":
        """A generation that did not happen, recorded rather than dropped.

        A failed cell is written with `finish_reason="error"` so that a resume
        can tell "attempted and failed" apart from "never attempted". Dropping
        it instead would make the sweep retry the same broken cell forever.
        """
        return RawGeneration(
            text="", model_id=model_id, finish_reason="error", error=error[:2000],
            mode=mode, batch_size=batch_size, wall_ms=wall_ms,
        )


class Backend(ABC):
    """Base class for generation engines.

    Subclasses implement `_generate` and `_resolve_model`. Everything shared —
    role resolution, per-request error containment, finish-reason
    normalization — lives here so a new backend cannot forget it.
    """

    #: Stable identifier recorded on every row this backend produces.
    name: str = "base"
    #: `sweep` (batched, wall-clock meaningless) or `serving` (fixed concurrency).
    mode: str = "sweep"
    #: Whether an explicit seed changes what this backend generates.
    #:
    #: False for hosted APIs that expose no seed parameter. It matters: at
    #: temperature 0 a backend that ignores seeds returns the *same* text for
    #: every seed, so a three-seed frozen ladder pays three times for one
    #: distinct generation and R4's cluster bootstrap sees zero within-task
    #: variance that is not actually zero. The sweep runner warns on it rather
    #: than letting it be discovered in the analysis.
    honors_seed: bool = True

    def __init__(self, small_model: str, large_model: str) -> None:
        self.small_model = small_model
        self.large_model = large_model

    def model_id(self, role: str) -> str:
        """Resolve a role to the model that will actually run."""
        if role == "small":
            return self.small_model
        if role == "large":
            return self.large_model
        raise ValueError(
            f"unknown model role {role!r}; arms name 'small' or 'large' and the "
            f"backend decides which weights back each rung"
        )

    def generate(self, requests: Sequence[GenRequest],
                 concurrency: int | None = None) -> list[RawGeneration]:
        """Generate for a batch of requests, in order.

        The returned list is positionally aligned with `requests`. A backend
        that fails wholesale still returns one `RawGeneration.failure` per
        request rather than raising, because a sweep that dies on a transient
        backend error loses hours of GPU time and resumes at the same cell.
        """
        if not requests:
            return []
        batch_size = len(requests) if self.mode == "sweep" else (concurrency or 1)
        t0 = time.perf_counter()
        try:
            results = list(self._generate(requests, batch_size))
        except Exception as exc:  # noqa: BLE001 - contained on purpose
            wall = (time.perf_counter() - t0) * 1000.0
            return [
                RawGeneration.failure(
                    self.model_id(r.model_role),
                    f"{exc.__class__.__name__}: {exc}",
                    mode=self.mode, batch_size=batch_size, wall_ms=wall,
                )
                for r in requests
            ]

        if len(results) != len(requests):
            raise BackendError(
                f"{self.name} returned {len(results)} results for "
                f"{len(requests)} requests; results must be positionally aligned"
            )
        return [self._finalize(r, batch_size) for r in results]

    def _finalize(self, raw: RawGeneration, batch_size: int) -> RawGeneration:
        """Normalize what a backend reported into the shared vocabulary.

        `mode` and `batch_size` are stamped here rather than trusted from the
        subclass, so a backend cannot accidentally label batched output as
        serving output and make its wall-clock look like latency.
        """
        reason = normalize_finish_reason(raw.finish_reason)
        return RawGeneration(
            text=raw.text, model_id=raw.model_id,
            prefill_tokens=raw.prefill_tokens, decode_tokens=raw.decode_tokens,
            wall_ms=raw.wall_ms, finish_reason=reason,
            mode=self.mode, batch_size=batch_size,
            tokens_exact=raw.tokens_exact, error=raw.error, extra=raw.extra,
        )

    @abstractmethod
    def _generate(self, requests: Sequence[GenRequest],
                  batch_size: int) -> Iterable[RawGeneration]:
        """Produce one result per request, in the order given."""

    def close(self) -> None:
        """Release engine resources. Safe to call more than once."""

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
