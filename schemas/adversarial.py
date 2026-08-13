"""Deliberately broken rollout stores.

Every case here is a real failure mode that produces *plausible numbers* rather
than an exception. That is the selection criterion: a corruption that crashes
the loader is not dangerous, because you find out. A corruption that shifts a
mean by two points and validates cleanly is the one that reaches a report.

Consumers are expected to reject these, not survive them. "Robust" here means
refusing to produce a number, not producing one anyway.

Each builder returns `(rows, why)` so a failing test can say what the fixture
was supposed to catch.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from .synth import SynthConfig, generate
from .version import SCHEMA_VERSION

Case = tuple[list[dict[str, Any]], str]

_SMALL = SynthConfig(n_tasks=40, seeds=3)


def _base() -> list[dict[str, Any]]:
    return copy.deepcopy(generate(_SMALL, seed=7).rows)


def mixed_schema_versions() -> Case:
    """Two contract versions in one store.

    The single most dangerous corruption, because every column still parses and
    every statistic still computes. Detectable only by checking the field that
    exists for this purpose.
    """
    rows = _base()
    for row in rows[: len(rows) // 3]:
        row["schema_version"] = SCHEMA_VERSION + 1
    return rows, "store mixes schema versions; analyses over it are invalid"


def duplicate_rollout_ids() -> Case:
    """The same cell present twice.

    Arises from a resume that lost its index, or two sweeps writing one run
    directory. Double-counts a task, tightening every interval.
    """
    rows = _base()
    rows.extend(copy.deepcopy(rows[:25]))
    return rows, "duplicate rollout_id double-counts cells and narrows intervals"


def split_leakage() -> Case:
    """One task appearing in two splits.

    Task-level splitting exists to prevent this. When it happens, the test set
    contains rows whose siblings trained the model, and the reported number is
    optimistic by an amount nobody can estimate after the fact.
    """
    rows = _base()
    victim = next(r["task_id"] for r in rows if r["split"] == "train")
    for row in rows:
        if row["task_id"] == victim and row["seed"] == 0:
            row["split"] = "test"
    return rows, "a task_id spans two splits; the test set is contaminated"


def ungraded_rows_as_zero() -> Case:
    """Half the store never graded.

    A consumer treating null as 0 reports a catastrophic accuracy drop and
    blames the model. A consumer that drops the rows silently reports a fine
    number over half the data.
    """
    rows = _base()
    for row in rows[::2]:
        row["hidden_passed"] = None
        row["hidden_total"] = None
        row["visible_passed"] = None
        row["visible_total"] = None
    return rows, "ungraded rows must not be read as zero-score or dropped silently"


def half_graded_pair() -> Case:
    """`passed` present, `total` missing.

    A truncated grading write. Any score computed from it divides by nothing.
    """
    rows = _base()
    for row in rows[:10]:
        row["hidden_total"] = None
    return rows, "passed without total is a partial write, not a zero"


def orphan_ladder_step() -> Case:
    """A repair row whose parent is absent.

    Breaks sequential replay: the ladder cannot be reconstructed, so any
    cascade or repair baseline computed over it is measuring a different
    policy than the one it claims.
    """
    rows = _base()
    row = copy.deepcopy(rows[0])
    row["ladder_step"] = 1
    row["parent_rollout_id"] = None
    rows.append(row)
    return rows, "ladder_step > 0 without a parent cannot be replayed"


def serving_row_batched() -> Case:
    """A latency measurement taken under batching.

    The exact category error the `mode`/`batch_size` pair exists to prevent.
    Produces a latency number that is really a queue-depth number.
    """
    rows = _base()
    for row in rows[:15]:
        row["mode"] = "serving"
        row["batch_size"] = 32
    return rows, "serving-mode latency under batching measures queue depth"


def negative_and_impossible_counts() -> Case:
    """`passed` greater than `total`, and a negative token count."""
    rows = _base()
    rows[0]["hidden_passed"] = rows[0]["hidden_total"] + 3
    rows[1]["decode_tokens"] = -5
    return rows, "impossible counts must be rejected, not clamped"


def unicode_and_nulls() -> Case:
    """Hostile text in fields that reach a report.

    Null bytes, RTL overrides, and lone surrogun-adjacent codepoints break
    naive serialization and can corrupt an HTML report.
    """
    rows = _base()
    rows[0]["code"] = "def f():\n    return '\u202e \ufffd'\n"
    rows[1]["task_id"] = "synth/\x00null"
    rows[2]["error_class"] = "\u03a9\u2248\u00e7\u221a\u222b"
    return rows, "hostile unicode must survive or be rejected, never corrupt output"


def single_arm_only() -> Case:
    """Only one arm was swept.

    Every paired comparison in the harness assumes both arms ran on every task.
    With one arm the paired test has no discordant pairs and will happily report
    a p-value of 1.0 rather than an error.
    """
    rows = [r for r in _base() if r["arm"] == "small"]
    return rows, "paired comparisons require both arms; one arm is not a tie"


def unsealed_run() -> Case:
    """Rows from a run whose manifest never landed.

    A sweep killed midway. The rows look complete because the ones that exist
    are complete; the missing ones are simply absent, biasing toward whichever
    tasks ran first.
    """
    rows = _base()[: len(_base()) // 2]
    return rows, "an unsealed run is partial; readers must skip it entirely"


def dirty_run_id() -> Case:
    """A run produced from an unclean worktree.

    Non-publishable: the recorded git sha does not describe the code that ran.
    """
    rows = _base()
    for row in rows:
        row["run_id"] = row["run_id"] + "-dirty"
    return rows, "-dirty runs are non-publishable"


CASES: dict[str, Callable[[], Case]] = {
    "mixed_schema_versions": mixed_schema_versions,
    "duplicate_rollout_ids": duplicate_rollout_ids,
    "split_leakage": split_leakage,
    "ungraded_rows_as_zero": ungraded_rows_as_zero,
    "half_graded_pair": half_graded_pair,
    "orphan_ladder_step": orphan_ladder_step,
    "serving_row_batched": serving_row_batched,
    "negative_and_impossible_counts": negative_and_impossible_counts,
    "unicode_and_nulls": unicode_and_nulls,
    "single_arm_only": single_arm_only,
    "unsealed_run": unsealed_run,
    "dirty_run_id": dirty_run_id,
}


def all_cases() -> dict[str, Case]:
    """Every adversarial fixture, built."""
    return {name: build() for name, build in CASES.items()}
