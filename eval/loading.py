"""Reading a rollout store, with every guard that stops a bad number.

This module is deliberately hostile to its input. Everything in
`schemas/adversarial.py` is a corruption that *validates cleanly and produces a
plausible number*, so the only place they can be caught is at the boundary —
here, once, loudly. A statistic cannot detect that it was computed over a
duplicated task or a mixed-version store; by the time the number exists the
evidence is gone.

The rule this module enforces above all others: **refusing to produce a number
is a correct outcome.** A loader that limps past a corrupted store and returns
something is worse than one that raises, because the former reaches a report.

The test split is guarded structurally rather than by convention. Reading it
requires an explicit unlock carrying a reason, and the unlock is recorded. See
`unlock_test_split`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from schemas import assert_single_version, iter_errors

MANIFEST_NAME = "_MANIFEST.json"

# Splits a consumer may read without ceremony. `test` is absent on purpose.
OPEN_SPLITS: tuple[str, ...] = ("train", "val")


class StoreError(ValueError):
    """The store cannot be read into a form any statistic should touch."""


class TestSplitError(StoreError):
    """Something tried to read the frozen test split without unlocking it."""

    # Not a pytest test class, despite the name. The name describes the domain
    # concept and is worth keeping; this tells pytest to stop trying to collect it.
    __test__ = False


@dataclass(frozen=True)
class TestSplitUnlock:
    """Permission to read the frozen test split, carrying its justification.

    Deliberately awkward to obtain. The discipline is that the test split is
    opened *once*, after the analysis is pre-registered, and an unlock that has
    to name a pre-registration file is one a person cannot produce by reflex
    while iterating.
    """

    __test__ = False  # see TestSplitError

    reason: str
    preregistration: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise TestSplitError("unlocking the test split requires a reason")
        if not self.preregistration.strip():
            raise TestSplitError(
                "unlocking the test split requires a path to the pre-registered "
                "analysis. Write down the metric, comparison, test, correction, "
                "and stopping rule first — then look."
            )


def unlock_test_split(*, reason: str, preregistration: str | Path) -> TestSplitUnlock:
    """Mint permission to read the frozen test split.

    `preregistration` must point at a file that already exists. Pre-registering
    after unblinding is not pre-registering, and a path checked at unlock time
    is the cheapest enforcement available.
    """
    path = Path(preregistration)
    if not path.exists():
        raise TestSplitError(
            f"pre-registration {path} does not exist. The analysis is written "
            f"and committed before the split is opened, not after."
        )
    return TestSplitUnlock(reason=reason, preregistration=str(path))


@dataclass
class Rollouts:
    """A validated, tidy view of one run's rollouts.

    Columns are numpy arrays of equal length, one element per row. Kept as
    arrays rather than a DataFrame so R3 and R4 need no dependency beyond numpy
    — the same reason they need no GPU.
    """

    columns: dict[str, np.ndarray]
    run_id: str
    splits: tuple[str, ...]
    arms: tuple[str, ...]
    n_tasks: int
    n_seeds: int
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.columns["task_id"])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    @property
    def task_ids(self) -> np.ndarray:
        return self.columns["task_id"]

    @property
    def solved(self) -> np.ndarray:
        """The headline label: every hidden test passed.

        Derived here, once, so no consumer invents its own definition. Partial
        credit is deliberately not the metric — a function that passes 7 of 8
        hidden tests is wrong.
        """
        return self.columns["_solved"]

    def arm_matrix(self, column: str) -> dict[str, np.ndarray]:
        """`column` reshaped to (n_tasks, n_seeds) per arm, task order fixed.

        This is the shape every paired comparison needs: the same tasks, in the
        same order, across arms. Building it here means no baseline can
        accidentally compare misaligned task orders — a bug that produces a
        plausible number and no error.
        """
        out: dict[str, np.ndarray] = {}
        order = {t: i for i, t in enumerate(self.ordered_tasks)}
        for arm in self.arms:
            mask = self.columns["arm"] == arm
            grid = np.full((self.n_tasks, self.n_seeds), np.nan, dtype=float)
            for task, seed, value in zip(
                self.columns["task_id"][mask],
                self.columns["seed"][mask],
                self.columns[column][mask],
            ):
                grid[order[task], int(seed)] = value
            out[arm] = grid
        return out

    @property
    def ordered_tasks(self) -> list[str]:
        """Task ids in a stable, sorted order.

        Sorted rather than first-seen: bootstrap reproducibility depends on the
        task index meaning the same thing across runs, and first-seen order
        depends on file iteration order.
        """
        return sorted(set(self.columns["task_id"].tolist()))


def load_rows(
    rows: Iterable[dict[str, Any]],
    *,
    splits: Sequence[str] = OPEN_SPLITS,
    unlock: TestSplitUnlock | None = None,
    require_graded: bool = True,
    require_complete_grid: bool = True,
    allow_dirty: bool = False,
) -> Rollouts:
    """Validate and tidy an in-memory rollout store.

    Every check here corresponds to an adversarial fixture. Raising is the
    expected behaviour for all of them.
    """
    rows = list(rows)
    if not rows:
        raise StoreError("empty store")

    # --- the contract itself ------------------------------------------------
    errors = [str(e) for e in iter_errors(rows)]
    if errors:
        raise StoreError(
            f"{len(errors)} rows violate the rollout contract; first 3:\n  "
            + "\n  ".join(errors[:3])
        )
    assert_single_version(rows)

    # --- one run, and a publishable one -------------------------------------
    run_ids = {r["run_id"] for r in rows}
    if len(run_ids) > 1:
        raise StoreError(
            f"store spans {len(run_ids)} run_ids {sorted(run_ids)[:3]}. Analyses "
            f"pin exactly one run_id; nothing reads 'latest'."
        )
    run_id = run_ids.pop()
    if run_id.endswith("-dirty") and not allow_dirty:
        raise StoreError(
            f"run {run_id} came from a dirty worktree, so the recorded git sha "
            f"does not describe the code that ran. Non-publishable. Pass "
            f"allow_dirty=True only for local iteration."
        )

    # --- duplicates ---------------------------------------------------------
    dupes = [k for k, n in Counter(r["rollout_id"] for r in rows).items() if n > 1]
    if dupes:
        raise StoreError(
            f"{len(dupes)} duplicate rollout_ids (e.g. {dupes[0]}). Duplicates "
            f"double-count tasks and narrow every interval."
        )

    # --- split integrity ----------------------------------------------------
    split_of: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_of[row["task_id"]].add(row["split"])
    straddling = [t for t, s in split_of.items() if len(s) > 1]
    if straddling:
        raise StoreError(
            f"{len(straddling)} task_ids appear in more than one split "
            f"(e.g. {straddling[0]}). Splitting is task-level; a straddling "
            f"task contaminates the test set by an amount nobody can estimate "
            f"after the fact."
        )

    # --- the frozen split ---------------------------------------------------
    requested = tuple(splits)
    if "test" in requested and unlock is None:
        raise TestSplitError(
            "reading the test split requires unlock_test_split(reason=..., "
            "preregistration=...). It is opened once, after the analysis is "
            "written down."
        )

    rows = [r for r in rows if r["split"] in requested]
    if not rows:
        raise StoreError(f"no rows in splits {requested}")

    # --- grading ------------------------------------------------------------
    ungraded = [r for r in rows if r["hidden_total"] is None]
    if ungraded and require_graded:
        raise StoreError(
            f"{len(ungraded)}/{len(rows)} rows are ungraded. Ungraded is not "
            f"zero-score and must not be silently dropped — grade the run, or "
            f"pass require_graded=False and handle the nulls explicitly."
        )

    # --- build the tidy view ------------------------------------------------
    return _tidy(rows, run_id, requested, require_complete_grid)


def _tidy(
    rows: list[dict[str, Any]],
    run_id: str,
    splits: tuple[str, ...],
    require_complete_grid: bool,
) -> Rollouts:
    arms = tuple(sorted({r["arm"] for r in rows}))
    tasks = sorted({r["task_id"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    if len(arms) < 2:
        raise StoreError(
            f"store has {len(arms)} arm(s) {arms}. Every comparison in this "
            f"harness is paired across arms; one arm is not a tie, it is a "
            f"missing experiment."
        )

    if require_complete_grid:
        expected = len(tasks) * len(arms) * len(seeds)
        if len(rows) != expected:
            missing = expected - len(rows)
            raise StoreError(
                f"incomplete grid: {len(rows)} rows, expected "
                f"{expected} = {len(tasks)} tasks x {len(arms)} arms x "
                f"{len(seeds)} seeds ({missing} missing). A hole in the grid "
                f"silently turns a paired test into an unpaired one."
            )
        if seeds != list(range(len(seeds))):
            raise StoreError(f"seeds are not contiguous from 0: {seeds}")

    columns: dict[str, np.ndarray] = {}
    for key, dtype in (
        ("task_id", object), ("arm", object), ("split", object),
        ("rollout_id", object), ("dataset", object),
    ):
        columns[key] = np.array([r[key] for r in rows], dtype=dtype)
    columns["seed"] = np.array([int(r["seed"]) for r in rows], dtype=int)

    for key in ("gpu_seconds", "imputed_latency_s", "wall_ms",
                "prefill_tokens", "decode_tokens"):
        columns[key] = np.array(
            [np.nan if r.get(key) is None else float(r[key]) for r in rows],
            dtype=float,
        )

    for key in ("visible_passed", "visible_total", "hidden_passed", "hidden_total"):
        columns[key] = np.array(
            [np.nan if r.get(key) is None else float(r[key]) for r in rows],
            dtype=float,
        )

    # The headline label, derived once. total > 0 guards a task with no hidden
    # tests, which would otherwise count as solved by vacuous truth.
    total = columns["hidden_total"]
    passed = columns["hidden_passed"]
    with np.errstate(invalid="ignore"):
        columns["_solved"] = ((total > 0) & (passed == total)).astype(float)
    columns["_solved"][np.isnan(total)] = np.nan

    # Visible pass fraction — the D1 feature. Never derived from hidden.
    with np.errstate(invalid="ignore", divide="ignore"):
        columns["_visible_frac"] = np.where(
            columns["visible_total"] > 0,
            columns["visible_passed"] / columns["visible_total"],
            np.nan,
        )

    columns["truncated"] = np.array(
        [r.get("finish_reason") == "length" for r in rows], dtype=float
    )
    columns["hacked"] = np.array(
        [bool(r.get("hack_flags")) for r in rows], dtype=float
    )

    warnings: list[str] = []
    if (rate := float(np.nanmean(columns["truncated"]))) > 0.02:
        warnings.append(
            f"truncation rate {rate:.1%}: 'length' finishes grade as capability "
            f"gaps but are failed generations. Investigate before reporting."
        )
    if (rate := float(np.nanmean(columns["hacked"]))) > 0.0:
        warnings.append(
            f"reward-hack flags on {rate:.1%} of rows. A rising rate is a "
            f"finding, not noise."
        )
    if np.isnan(columns["gpu_seconds"]).any():
        warnings.append(
            "some rows have no gpu_seconds: the characterization pass has not "
            "been imputed onto this run. Cost comparisons are unavailable."
        )

    return Rollouts(
        columns=columns,
        run_id=run_id,
        splits=splits,
        arms=arms,
        n_tasks=len(tasks),
        n_seeds=len(seeds),
        warnings=warnings,
    )


def load_run(
    root: str | Path,
    run_id: str,
    *,
    sealed_only: bool = True,
    **kwargs: Any,
) -> Rollouts:
    """Read one sealed run directory from disk.

    A run is invalid until `_MANIFEST.json` lands. Skipping unsealed runs is
    what stops a half-finished sweep being read as a complete one — the rows
    that exist are complete, so nothing about them looks wrong.
    """
    run_dir = Path(root) / run_id
    if not run_dir.is_dir():
        raise StoreError(f"no run directory at {run_dir}")
    if sealed_only and not (run_dir / MANIFEST_NAME).exists():
        raise StoreError(
            f"run {run_id} has no {MANIFEST_NAME} and is not valid to read. "
            f"An unsealed run is partial, and its missing rows are biased "
            f"toward whichever tasks ran last."
        )

    rows: list[dict[str, Any]] = []
    for part in sorted((run_dir / "generations").glob("part-*.jsonl")):
        with open(part, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise StoreError(f"run {run_id} contains no rows")
    return load_rows(rows, **kwargs)
