"""Laying `schemas.synth` out on disk so the real read path consumes it.

The generator produces rows in memory. R3's loader reads a *run directory* —
part files, a manifest, a cost sidecar, a task manifest. This module is the
adapter between them, and it exists so that the switch from fixtures to the
real store in week 2 is a path change and nothing else. Every test in this
package that touches data goes through `load_rollouts`, exactly as production
code will, rather than through a list of dicts a real run would never produce.

## The D0 problem this module has to solve

The generator plants a prompt-only proxy for difficulty, `x_d0`. It is the D0
signal — the thing a pre-generation router would have to work from. But it
appears in exactly two places: `SynthResult.truth`, which is ground truth and
not observable, and `extra["_synth_x_d0"]`, which R3's loader quarantines
because `extra["_synth_difficulty"]` sits beside it and reading *that* is
leakage.

So on the fixture as generated, **there is no legitimate D0 feature surface at
all**, and `AUC_D0` — a Phase 0 gate number — cannot be computed without
reaching into ground truth.

That is not a flaw in the generator; it is the prompt gap showing up again. The
rollout row carries no prompt, in the ratified schema as much as in R1's draft.
Real D0 features come from R2's task manifest, joined by `task_id`. So the
fixture supplies a task manifest too, and `x_d0` arrives through exactly the
channel a real prompt feature would:

* `task_x_d0` — the proxy itself, on the clean side of the join. A fixture-only
  affordance, and named so nobody mistakes it for something a real corpus has.
* `task_prompt` — real prose whose **length is a monotone function of `x_d0`**,
  so a genuine prompt-length feature recovers the planted signal without any
  knowledge of the fixture. This is the one that matters: it means Phase 3's
  feature builders are exercised end to end rather than handed the answer.

`_synth_difficulty` stays quarantined in `.latent` throughout. Nothing here
gives a feature path access to the latent value, only to the noisy proxy a real
prompt would carry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from schemas.synth import SynthConfig, SynthResult, generate

#: Filenames inside a run directory. Kept identical to R1's layout — a fixture
#: that is laid out differently from a real run is a fixture that tests a code
#: path production never takes.
ROWS_DIR = "generations"
MANIFEST_NAME = "_MANIFEST.json"
COST_DIR = "cost"

#: Word count of the shortest synthetic prompt, and the span above it that
#: `x_d0` is mapped onto. Wide enough that the length ordering survives being
#: rounded to whole words, which is what a real tokenizer would do to it.
_MIN_PROMPT_WORDS = 12
_PROMPT_WORD_SPAN = 160

_FILLER = (
    "the function should handle the empty case and preserve input order "
    "while avoiding an allocation in the inner loop for large inputs "
).split()


@dataclass(frozen=True)
class Fixture:
    """A synthetic run on disk, plus the ground truth it was built from."""

    root: Path
    run_id: str
    tasks_path: Path
    result: SynthResult
    cost_fingerprint: str | None = None

    @property
    def truth(self) -> dict[str, Any]:
        return self.result.truth

    @property
    def config(self) -> SynthConfig:
        return self.result.truth["config"]

    def planted_auc_d0(self) -> float:
        """The ceiling any D0 model on this fixture can reach.

        A feature builder reporting more than this has leaked something, which
        makes it an assertion rather than a target.
        """
        return self.result.planted_auc_d0()


def _prompt_for(x_d0: float, lo: float, hi: float, task_id: str) -> str:
    """Prose whose length encodes the D0 proxy, **inversely**.

    Monotone in `x_d0` by construction, so a prompt-length feature is a valid
    D0 signal on this fixture in the same way it is on a real corpus — without
    the feature builder knowing the fixture exists.

    The sign matters and is chosen deliberately. `x_d0` is oriented so that
    higher means *more likely to solve*; a longer prompt, in a real corpus,
    means a *harder* task. Mapping length directly onto `x_d0` would therefore
    plant a relationship whose direction is backwards from the one a real store
    has, and a feature builder validated against it would carry that inversion
    into week 4 — where the fix looks like flipping a sign, which is fitting
    the label rather than the feature.

    So length runs against the proxy: high `x_d0` (easy) produces a short
    prompt.
    """
    span = (hi - lo) or 1.0
    scaled = 1.0 - (x_d0 - lo) / span
    n_words = _MIN_PROMPT_WORDS + int(round(scaled * _PROMPT_WORD_SPAN))
    words = [(_FILLER[i % len(_FILLER)]) for i in range(n_words)]
    return f"Implement the routine described for {task_id}: " + " ".join(words) + "."


def write_tasks(path: Path, result: SynthResult) -> Path:
    """Write the task manifest the D0 features are joined from.

    Shaped like R2's corpus so `read_task_features` reads it unmodified: one
    JSON object per line, `task_id` plus prompt-side fields. The hidden suite is
    absent rather than filtered — there is nothing here for the join to leave
    behind, which is the strongest form of the guarantee.
    """
    x_d0 = result.truth["x_d0"]
    task_ids: Sequence[str] = result.truth["task_ids"]
    lo, hi = float(min(x_d0)), float(max(x_d0))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        for i, task_id in enumerate(task_ids):
            value = float(x_d0[i])
            record = {
                "task_id": task_id,
                "prompt": _prompt_for(value, lo, hi, task_id),
                "entrypoint": "solve",
                "visible_tests": "\n".join(
                    f"assert solve({k}) is not None"
                    for k in range(result.truth["config"].visible_tests)
                ),
                "metadata": {"x_d0": value},
            }
            fh.write(json.dumps(record) + "\n")
    return path


def write_cost_sidecar(root: Path, run_id: str, rows: list[dict[str, Any]],
                       fingerprint: str = "5ynth1c0") -> str:
    """Write a costing beside the rows, keyed by fingerprint like a real one.

    The values are copied from the rows the generator already priced rather
    than invented, so a test that pins this costing and one that reads the
    row's own `gpu_seconds` agree — a disagreement there would be a bug in the
    fixture masquerading as a bug in the join.
    """
    directory = root / run_id / COST_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{fingerprint}.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({
                "rollout_id": row["rollout_id"],
                "task_id": row["task_id"],
                "arm": row["arm"],
                "seed": row["seed"],
                "model_id": row["model_id"],
                "gpu_seconds": row["gpu_seconds"],
                "imputed_latency_s": row["imputed_latency_s"],
                "usd": row["gpu_seconds"] * 1.10 / 3600.0,
            }) + "\n")
    return fingerprint


#: R2's graded layer, restated here for the same reason `store.py` restates it:
#: `graders.rollout_store` is not on `main` yet.
GRADED_ROWS_DIR = "rollouts"
GRADED_MANIFEST_NAME = "_ROLLOUT_MANIFEST.json"

#: Grading columns R2 fills. Nulled out in the generations layer so a `split`
#: layout reproduces the real thing: R1 writes these as nulls, R2 fills them in
#: a separate directory, and nothing is ever mutated in place.
_GRADE_COLUMNS = (
    "visible_passed", "visible_total", "hidden_passed", "hidden_total",
    "error_class", "hack_flags", "grade_duration_s",
)


def _write_part(directory: Path, rows: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    part = directory / "part-000001-0000000000-5ynth1.jsonl"
    with part.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    return part


def write_fixture(root: Path | str, config: SynthConfig | None = None, *,
                  seed: int = 0, sealed: bool = True,
                  with_cost: bool = True,
                  layout: str = "generations",
                  seal_graded: bool = True) -> Fixture:
    """Materialize a synthetic run as a readable run directory.

    `layout` picks the shape on disk:

    * `"generations"` — one layer, grades already present. Compact, and what
      most tests want.
    * `"split"` — the real production shape: R1's ungraded rows in
      `generations/`, the same rows with grades filled in `rollouts/`, each
      sealed by its own manifest. R2 never edits R1's files, because a sealed
      run's manifest carries per-file checksums.

    `sealed=False` produces the one shape a reader must skip: rows on disk with
    no manifest. It is not a broken fixture, it is an interrupted sweep, and
    the distinction is the whole reason the seal exists. `seal_graded=False`
    does the same one layer up — grading interrupted half way.
    """
    if layout not in ("generations", "split"):
        raise ValueError(
            f"unknown layout {layout!r}; expected 'generations' or 'split'"
        )

    root = Path(root)
    result = generate(config, seed=seed)
    run_id = result.run_id
    directory = root / run_id

    if layout == "split":
        ungraded = [
            {**row, **{column: None for column in _GRADE_COLUMNS}}
            for row in result.rows
        ]
        _write_part(directory / ROWS_DIR, ungraded)
        _write_part(directory / GRADED_ROWS_DIR, result.rows)
    else:
        _write_part(directory / ROWS_DIR, result.rows)

    fingerprint = (write_cost_sidecar(root, run_id, result.rows)
                   if with_cost else None)
    tasks_path = write_tasks(directory / "tasks.jsonl", result)

    if sealed:
        (directory / MANIFEST_NAME).write_text(json.dumps({
            "run_id": run_id,
            "schema_version": result.rows[0]["schema_version"],
            "sealed_at": f"{result.truth['config'].date}T00:00:00Z",
            "publishable": not run_id.endswith("-dirty"),
            "n_rows": len(result.rows),
            "source": "schemas.synth",
        }, indent=2, sort_keys=True), encoding="utf-8")

    if layout == "split" and seal_graded:
        # Deliberately carries no `publishable` key, matching R2's seal. A
        # fixture that added one would hide the fact that publishability has
        # to come from R1's manifest.
        (directory / GRADED_MANIFEST_NAME).write_text(json.dumps({
            "run_id": run_id,
            "sealed_at": f"{result.truth['config'].date}T00:00:00Z",
            "n_rows": len(result.rows),
            "solved_count": sum(
                1 for row in result.rows
                if row["hidden_total"] and row["hidden_passed"] == row["hidden_total"]
            ),
            "source": "schemas.synth",
        }, indent=2, sort_keys=True), encoding="utf-8")

    return Fixture(
        root=root,
        run_id=run_id,
        tasks_path=tasks_path,
        result=result,
        cost_fingerprint=fingerprint,
    )
