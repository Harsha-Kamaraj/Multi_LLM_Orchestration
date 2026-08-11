"""The `Generation` record — R1's output, and the row every other role reads.

One `Generation` is one `(task_id, arm, seed)` cell. It is written once, never
mutated, and projects directly onto the shared rollout schema with R2's grading
fields left null for R2 to fill.

Two fields exist purely to stop a category error:

* **`mode`** — `sweep` or `serving`.
* **`batch_size`** — how many requests shared the call.

A batched sweep runs many requests concurrently, so any one request's
wall-clock is a function of queue depth rather than of the model. `wall_ms` is
still recorded, because it is a useful sanity signal, but a consumer that wants
latency must filter to `mode == "serving"`. Carrying both fields makes that
filter possible; omitting them would make sweep wall-clock *look* like latency,
and the resulting error is invisible — it produces plausible numbers.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Bumped on any destructive change to this record. Every row carries it so a
# mixed-version store is detectable rather than silently averaged.
SCHEMA_VERSION = 1

Mode = Literal["sweep", "serving"]

# Normalized across backends. Each one reports something different, and
# collapsing them at the edge means nothing downstream has to know which
# backend produced a row.
FinishReason = Literal["stop", "length", "refusal", "error", "unknown"]
FINISH_REASONS: tuple[str, ...] = ("stop", "length", "refusal", "error", "unknown")

_FINISH_ALIASES: dict[str, str] = {
    # vLLM / OpenAI-compatible
    "stop": "stop",
    "length": "length",
    "abort": "error",
    "content_filter": "refusal",
    "tool_calls": "stop",
    # Anthropic
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "refusal",
    "tool_use": "stop",
    "pause_turn": "stop",
    "model_context_window_exceeded": "length",
}


def normalize_finish_reason(raw: str | None) -> str:
    """Map a backend's finish reason onto the shared vocabulary.

    An unrecognised value becomes `"unknown"` rather than being passed through.
    A new value from a backend upgrade should show up as a visible bucket in
    the manifest, not as a new string nobody is counting.
    """
    if raw is None:
        return "unknown"
    return _FINISH_ALIASES.get(str(raw).strip().lower(), "unknown")


@dataclass(frozen=True)
class Generation:
    """One generated candidate, with full accounting.

    Frozen. Imputed cost fields are attached with `with_cost`, which returns a
    new record — a row is never mutated in place, and the store enforces the
    same thing at the file level.
    """

    # --- identity -----------------------------------------------------------
    run_id: str
    task_id: str
    arm: str
    seed: int
    params_hash: str

    # --- generation ---------------------------------------------------------
    text: str = ""
    code: str = ""
    model_id: str = ""
    prefill_tokens: int = 0
    decode_tokens: int = 0
    wall_ms: float = 0.0
    finish_reason: str = "unknown"

    # --- how it was produced ------------------------------------------------
    # Together these are what stop sweep wall-clock being read as latency.
    mode: str = "sweep"
    batch_size: int = 1
    backend: str = ""
    # False when token counts came from an approximation rather than from the
    # serving tokenizer. Any row with this false is unfit for cost fitting.
    tokens_exact: bool = True

    # --- extraction (R1 owns this; R2 receives `code`) -----------------------
    extract_strategy: str = ""
    code_parses: bool = False

    # --- corpus context, passed through from R2's manifest -------------------
    split: str = ""
    dataset: str = ""
    code_version: str = ""

    # --- repair ladder ------------------------------------------------------
    ladder_step: int = 0
    parent_rollout_id: str | None = None

    # --- imputed by the characterization pass, not measured here ------------
    gpu_seconds: float | None = None
    imputed_latency_s: float | None = None

    # --- provenance ---------------------------------------------------------
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rollout_id(self) -> str:
        """Deterministic identity for this cell.

        Derived rather than random so a re-run under identical conditions
        produces the same id, and so a child generation can reference its
        parent without the parent having been written yet.
        """
        h = hashlib.sha256()
        for part in (
            self.run_id, self.task_id, self.arm, str(self.seed),
            self.params_hash, str(self.ladder_step),
            self.parent_rollout_id or "",
        ):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    @property
    def cell_key(self) -> tuple[str, str, int, str]:
        """The resume key: `(task_id, arm, seed, params_hash)`.

        `params_hash` is in the key so a cell computed under different
        parameters is never mistaken for a hit. Without it, resuming after a
        prompt tweak would keep the stale rows and produce a store containing
        two different experiments.
        """
        return (self.task_id, self.arm, self.seed, self.params_hash)

    @property
    def truncated(self) -> bool:
        """A failed generation dressed as a successful one.

        `finish_reason == "length"` grades as a failure and looks exactly like
        a model capability gap. The rate is reported in every manifest.
        """
        return self.finish_reason == "length"

    @property
    def failed(self) -> bool:
        """Whether the backend never produced usable output."""
        return self.finish_reason in ("error", "refusal") or not self.code.strip()

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens

    def with_cost(self, gpu_seconds: float, imputed_latency_s: float) -> "Generation":
        """Return a copy carrying imputed cost. Never mutates."""
        return Generation(**{
            **asdict(self),
            "gpu_seconds": gpu_seconds,
            "imputed_latency_s": imputed_latency_s,
        })

    def to_row(self) -> dict[str, Any]:
        """Project onto the shared rollout schema.

        R2's grading fields are present and null. Emitting the keys rather than
        omitting them means a store read before grading has the same columns as
        one read after, so nothing downstream has to special-case an ungraded
        run.
        """
        row = asdict(self)
        row["rollout_id"] = self.rollout_id
        row.update({
            "visible_passed": None,
            "visible_total": None,
            "hidden_passed": None,
            "hidden_total": None,
            "error_class": None,
            "hack_flags": None,
            "grade_duration_s": None,
        })
        return row

    @staticmethod
    def from_row(row: dict[str, Any]) -> "Generation":
        """Rebuild from a stored row, ignoring fields owned by other roles."""
        fields = {f for f in Generation.__dataclass_fields__}
        return Generation(**{k: v for k, v in row.items() if k in fields})
