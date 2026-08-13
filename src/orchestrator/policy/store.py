"""Reading a pinned run out of the rollout store.

Everything R3 builds stands on this module, so it is deliberately strict. Four
refusals, each protecting a number that would otherwise look plausible:

**Unsealed runs.** A run is invalid until `_MANIFEST.json` lands. Reading one
anyway produces a real accuracy computed over two thirds of a corpus.

**The test split.** There is no flag for it. R4 opens it exactly once, after
pre-registration, and the reason R3 and R4 are two people evaporates the moment
R3 can reach it by passing `True` somewhere. A code change with a reviewer is a
higher bar than an argument, and that is the point.

**Ungraded runs.** Every real row today has null grading columns — R1 writes
them as nulls for R2 to fill, and no grader writes them back yet. Training on
that silently yields an all-negative label column and an AUC of 0.5 that reads
as a modelling failure rather than as missing data.

**"Latest".** Nothing here resolves one. A `run_id` is pinned by the caller,
and a costing is pinned by coefficient fingerprint the same way — two costings
of one run legitimately coexist, so "the cost" is not a well-formed request.

## Two read paths, one result

Parquet is what R3 and R4 read; JSONL is authoritative and is what exists when
the sweep host had no pyarrow. Both are supported, both go through
`contract.normalize_row`, and `test_store.py` asserts the two agree row for
row. Without that test the paths drift, and the symptom is a result that
reproduces on one machine and not another.

## The hidden columns never make it out of this module

`load_rollouts` returns rows with `hidden_passed` and `hidden_total` removed
and hands the labels back separately, keyed by `rollout_id`. A feature builder
receives `.rows` and is never given `.labels`. That is a structural guard: not
a rule anyone has to remember, and not one a new code path can quietly skip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from schemas.validate import rollout_schema

from ..workers.store import MANIFEST_NAME, PARQUET_NAME, ROWS_DIR
from . import contract
from .errors import SplitError, StoreReadError, UngradedRunError

#: Where R1's imputation pass writes cost sidecars, one per coefficient set.
COST_DIR = "cost"

#: The split vocabulary, read out of the ratified schema itself.
_SCHEMA_SPLITS: tuple[str, ...] = tuple(
    rollout_schema()["properties"]["split"]["enum"]
)

#: The splits R3 is entitled to. `test` is absent, and its absence is the
#: enforcement — see the module docstring.
#:
#: Derived from the schema's own enum rather than written out, so a contract
#: change that adds a split cannot leave this list quietly stale. Only `test`
#: is subtracted, and only here.
ALLOWED_SPLITS: frozenset[str] = frozenset(_SCHEMA_SPLITS) - {"test"}

#: The only fields that cross from R2's task manifest into a feature path.
#: `Task.tests` holds the full suite and is left behind; so is any reference
#: solution a corpus may carry. R1's `PromptContext` makes the same cut for the
#: same reason — a template that cannot reach hidden tests cannot render them.
TASK_FIELDS: tuple[str, ...] = ("task_prompt", "task_entrypoint",
                                "task_visible_tests")


@dataclass(frozen=True)
class Label:
    """One hidden-test outcome. The target, and never an input."""

    rollout_id: str
    task_id: str
    arm: str
    seed: int
    hidden_passed: int
    hidden_total: int

    @property
    def solved(self) -> bool:
        """Strict all-or-nothing correctness — what the project reports."""
        return self.hidden_total > 0 and self.hidden_passed == self.hidden_total


@dataclass(frozen=True)
class RolloutData:
    """One pinned run, split into what a model may see and what it predicts.

    `rows` and `labels` are separate objects on purpose. Handing a feature
    builder the whole thing and asking it not to look would be a convention;
    handing it `rows`, which does not contain the columns, is a guarantee.
    """

    run_id: str
    rows: tuple[dict[str, Any], ...]
    labels: Mapping[str, Label]
    manifest: dict[str, Any]
    source: str
    cost_fingerprint: str | None = None
    tasks_path: str | None = None

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def publishable(self) -> bool:
        """Whether anything derived from this run may be published.

        A run swept from a dirty worktree is stamped `-dirty` and is not
        publishable: the recorded git sha does not describe the code that
        produced the rows. Developing against one is fine; reporting from one
        is not, and Phase 6 stamps the decision file accordingly.
        """
        return bool(self.manifest.get("publishable", False))

    @property
    def has_cost(self) -> bool:
        return self.cost_fingerprint is not None

    def column(self, name: str) -> list[Any]:
        return [row.get(name) for row in self.rows]

    def unique(self, name: str) -> tuple[Any, ...]:
        seen: dict[Any, None] = {}
        for row in self.rows:
            seen.setdefault(row.get(name), None)
        return tuple(seen)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(str(t) for t in self.unique("task_id"))

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(str(a) for a in self.unique("arm"))

    def label_for(self, rollout_id: str) -> Label:
        try:
            return self.labels[rollout_id]
        except KeyError:
            raise StoreReadError(
                f"no label for rollout {rollout_id!r}; it was either filtered "
                f"out by the split selection or its run is ungraded"
            ) from None

    def solved_by_rollout(self) -> dict[str, bool]:
        """The label column, keyed the same way the rows are."""
        return {rid: label.solved for rid, label in self.labels.items()}

    def counts_by_split(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            key = str(row.get("split") or "unassigned")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def counts_by_arm(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            key = str(row.get("arm"))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def run_dir(root: Path | str, run_id: str) -> Path:
    return Path(root) / run_id


def cost_dir(root: Path | str, run_id: str) -> Path:
    return run_dir(root, run_id) / COST_DIR


def list_cost_fingerprints(root: Path | str, run_id: str) -> list[str]:
    """Every costing available for a run.

    Plural on purpose. The A100 costing and the H100 costing of the same
    generations coexist by design, so a caller that wants cost names which one
    it means — exactly as it names a `run_id`.
    """
    directory = cost_dir(root, run_id)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.jsonl"))


def read_manifest(root: Path | str, run_id: str) -> dict[str, Any]:
    """Load the seal, or refuse the run because it has none."""
    directory = run_dir(root, run_id)
    if not directory.exists():
        raise StoreReadError(
            f"no such run: {directory}. Nothing resolves 'latest' — pin an "
            f"explicit run_id from `orch-workers runs`."
        )
    path = directory / MANIFEST_NAME
    if not path.exists():
        raise StoreReadError(
            f"run {run_id} has no {MANIFEST_NAME} and is not valid to read; it "
            f"is either still running or was interrupted. Readers skip "
            f"unsealed runs, which is what stops a partial sweep being read as "
            f"a complete one."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The two read paths
# ---------------------------------------------------------------------------


def _iter_jsonl(root: Path | str, run_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream rows from the authoritative part files, with line numbers.

    A torn final line is skipped rather than raised on: a sweep killed
    mid-write leaves one, and it contributes nothing — resume regenerates that
    cell. Every other parse failure is a real problem and is reported with its
    file and line, because a bare decode error on a six-thousand-row store is
    close to useless.
    """
    rows_dir = run_dir(root, run_id) / ROWS_DIR
    parts = sorted(rows_dir.glob("part-*.jsonl"))
    if not parts:
        raise StoreReadError(f"run {run_id} has no part files under {rows_dir}")
    for part in parts:
        with part.open("r", encoding="utf-8", newline="") as fh:
            lines = fh.readlines()
        for lineno, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if lineno == len(lines):
                    continue  # torn tail of an interrupted sweep
                raise StoreReadError(
                    f"{part.name}:{lineno}: not valid JSON, and it is not the "
                    f"final line — this run is damaged rather than interrupted"
                ) from None
            yield f"{part.name}:{lineno}", row


def _iter_parquet(root: Path | str, run_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream rows from the derived Parquet view."""
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    path = run_dir(root, run_id) / ROWS_DIR / PARQUET_NAME
    table = pq.read_table(path)
    for index, row in enumerate(table.to_pylist()):
        yield f"{PARQUET_NAME}:{index}", row


def _parquet_available(root: Path | str, run_id: str) -> bool:
    path = run_dir(root, run_id) / ROWS_DIR / PARQUET_NAME
    if not path.exists():
        return False
    try:
        import pyarrow.parquet  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def read_cost_sidecar(root: Path | str, run_id: str,
                      fingerprint: str) -> dict[str, dict[str, Any]]:
    """Load one costing, keyed by `rollout_id`.

    A pinned fingerprint that does not exist raises and lists what does. The
    alternative — falling back to another costing — produces cost numbers that
    look fine and describe different hardware.
    """
    path = cost_dir(root, run_id) / f"{fingerprint}.jsonl"
    if not path.exists():
        available = list_cost_fingerprints(root, run_id)
        detail = (
            f"available costings are {available}" if available else
            "nothing has been imputed for this run, which needs "
            "bench/cost_coefficients.json from R1's characterization pass"
        )
        raise StoreReadError(
            f"run {run_id} has no cost sidecar {fingerprint!r}; {detail}"
        )
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[str(entry["rollout_id"])] = entry
    return out


def read_task_features(tasks_path: Path | str) -> dict[str, dict[str, str]]:
    """Load the prompt-side fields R2's manifest contributes, and nothing else.

    The rollout row carries no prompt — it carries what the model produced and
    the `task_id` that produced it — so every prompt-only D0 feature needs this
    join. Exactly three fields cross it. `Task.tests` is the full suite and
    stays behind; so does any reference solution the corpus may carry, which
    would be a label wearing the most obvious disguise available.
    """
    from ..workers.corpus import load_tasks

    tasks = load_tasks(tasks_path)
    out: dict[str, dict[str, str]] = {}
    for task in tasks:
        out[task.task_id] = {
            "task_prompt": task.prompt,
            "task_entrypoint": task.entrypoint,
            "task_visible_tests": str(task.metadata.get("visible_tests", "")),
        }
    return out


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------


def _check_splits(splits: Sequence[str]) -> frozenset[str]:
    wanted = frozenset(str(s) for s in splits)
    if not wanted:
        raise SplitError("no splits requested; name at least one of "
                         f"{sorted(ALLOWED_SPLITS)}")
    forbidden = sorted(wanted - ALLOWED_SPLITS)
    if forbidden:
        raise SplitError(
            f"R3 may not load {forbidden}. The test split is R4's, is opened "
            f"exactly once after the analysis is pre-registered, and is not "
            f"reachable through this package — there is no flag, by design. "
            f"Loadable splits: {sorted(ALLOWED_SPLITS)}."
        )
    return wanted


def load_rollouts(root: Path | str, run_id: str, *,
                  splits: Sequence[str] = ("train", "val"),
                  cost_fingerprint: str | None = None,
                  tasks_path: Path | str | None = None,
                  require_grades: bool = True,
                  prefer_parquet: bool = True) -> RolloutData:
    """Load one pinned run as features and labels.

    Args:
        root: the store root, conventionally `runs/`.
        run_id: pinned explicitly. Nothing here resolves "latest".
        splits: which splits to load. `test` is refused.
        cost_fingerprint: which costing to attach. `None` attaches none, which
            is the honest default while no coefficients exist.
        tasks_path: R2's manifest. Without it the prompt-side D0 features
            cannot be built, because the rollout row carries no prompt.
        require_grades: refuse a run R2 has not graded. Leave this on unless
            you are inspecting a sweep rather than training on one.
    """
    wanted_splits = _check_splits(splits)
    manifest = read_manifest(root, run_id)

    use_parquet = prefer_parquet and _parquet_available(root, run_id)
    reader = _iter_parquet if use_parquet else _iter_jsonl
    source = "parquet" if use_parquet else "jsonl"

    costs = (read_cost_sidecar(root, run_id, cost_fingerprint)
             if cost_fingerprint is not None else {})
    task_features = (read_task_features(tasks_path)
                     if tasks_path is not None else {})

    rows: list[dict[str, Any]] = []
    labels: dict[str, Label] = {}
    seen_versions: set[int] = set()
    n_total = 0
    n_ungraded = 0

    for where, raw in reader(root, run_id):
        n_total += 1
        row = contract.normalize_row(raw, where=where)
        seen_versions.add(int(row["schema_version"]))

        if str(row.get("split") or "") not in wanted_splits:
            continue

        rollout_id = str(row["rollout_id"])

        if contract.is_graded(row):
            labels[rollout_id] = Label(
                rollout_id=rollout_id,
                task_id=str(row["task_id"]),
                arm=str(row["arm"]),
                seed=int(row["seed"]),
                hidden_passed=int(row["hidden_passed"]),
                hidden_total=int(row["hidden_total"]),
            )
        else:
            n_ungraded += 1

        # The strip. From here on the row physically cannot leak a label,
        # whatever any downstream code decides to do with it.
        for column in contract.LABEL_COLUMNS:
            row.pop(column, None)

        if costs:
            priced = costs.get(rollout_id)
            if priced is not None:
                row["gpu_seconds"] = priced.get("gpu_seconds")
                row["imputed_latency_s"] = priced.get("imputed_latency_s")
                row["usd"] = priced.get("usd")

        if task_features:
            joined = task_features.get(str(row["task_id"]))
            if joined is None:
                raise StoreReadError(
                    f"{where}: task {row['task_id']!r} is in the run but not in "
                    f"the manifest at {tasks_path}. The corpus does not match "
                    f"the run — a sweep folds a corpus fingerprint into its "
                    f"run_id precisely so this mismatch is visible."
                )
            row.update(joined)

        rows.append(row)

    if len(seen_versions) > 1:
        from .errors import SchemaVersionError
        raise SchemaVersionError(
            f"run {run_id} mixes schema versions {sorted(seen_versions)}. A "
            f"mixed-version store must be detectable, never silently averaged: "
            f"two rows whose columns mean different things are two experiments."
        )

    if not rows:
        raise StoreReadError(
            f"run {run_id} has {n_total} rows but none in splits "
            f"{sorted(wanted_splits)}. Splits present: "
            f"{sorted({str(r.get('split') or '') for _, r in reader(root, run_id)})}"
        )

    if require_grades and n_ungraded:
        raise UngradedRunError(
            f"run {run_id}: {n_ungraded} of {len(rows)} selected rows have no "
            f"hidden-test outcome, so they carry no label. R1 writes the "
            f"grading columns as nulls for R2 to fill; a run in that state is "
            f"a valid sweep and an unusable training set. Pass "
            f"require_grades=False only to inspect one."
        )

    contract.assert_no_labels(
        {column for row in rows for column in row},
        context=f"rows loaded from run {run_id}",
    )

    return RolloutData(
        run_id=run_id,
        rows=tuple(rows),
        labels=labels,
        manifest=manifest,
        source=source,
        cost_fingerprint=cost_fingerprint,
        tasks_path=str(tasks_path) if tasks_path is not None else None,
    )
