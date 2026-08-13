"""Corruptions that produce a plausible number rather than an exception.

The selection criterion is R4's, from `schemas/adversarial.py`, and it is the
right one: a corruption that crashes the loader is not dangerous, because you
find out. A corruption that shifts a mean by two points and validates cleanly
is the one that reaches a report.

So every check here refuses to produce a number rather than producing one
anyway. All checks run before anything is thrown away, and **all issues are
reported together** — fixing a store one exception at a time, re-running a
sweep between each, is how a morning disappears.

Two checks deserve their reasoning stated, because both look pedantic until
they cost you a week:

**A task spanning two splits** is checked across *every* split, including the
one R3 may not load. Filtering to train and val first would hide exactly the
contamination the check exists to find — the train rows would look fine on
their own, and the leak would live in the half of the store this role never
sees.

**A single arm** is not a degenerate case to tolerate. The entire question is
which arm to choose; with one arm there is no choice, and every downstream
comparison silently becomes a comparison of a thing with itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: Identity fields where a control character is fatal rather than cosmetic.
#: A null byte in a `task_id` breaks the join key, the split manifest, and any
#: path derived from it — and does so silently, because most of the string
#: still renders.
_IDENTITY_FIELDS: tuple[str, ...] = ("task_id", "arm", "split", "run_id",
                                     "rollout_id")

#: How many offending rows to name in a message. Enough to grep for, few
#: enough that the error stays readable.
_MAX_EXAMPLES = 3


@dataclass(frozen=True)
class IntegrityIssue:
    """One thing wrong with a store, and how much of it is affected."""

    check: str
    detail: str
    n_rows: int
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        line = f"[{self.check}] {self.detail} ({self.n_rows} rows)"
        if self.examples:
            line += f"; e.g. {', '.join(self.examples)}"
        return line


def _examples(items: Iterable[Any]) -> tuple[str, ...]:
    out = []
    for item in items:
        out.append(str(item))
        if len(out) >= _MAX_EXAMPLES:
            break
    return tuple(out)


def _has_control_chars(value: Any) -> bool:
    return isinstance(value, str) and any(ord(c) < 0x20 for c in value)


def duplicate_rollout_ids(rows: Sequence[Mapping[str, Any]]) -> IntegrityIssue | None:
    """The same cell present twice.

    A resume that lost its index, or two sweeps writing one run directory.
    Double-counts a task, which tightens every interval it touches.
    """
    seen: dict[str, int] = {}
    for row in rows:
        key = str(row.get("rollout_id"))
        seen[key] = seen.get(key, 0) + 1
    dupes = {k: n for k, n in seen.items() if n > 1}
    if not dupes:
        return None
    return IntegrityIssue(
        check="duplicate_rollout_id",
        detail="the same cell appears more than once, which double-counts it "
               "and narrows every interval computed over it",
        n_rows=sum(dupes.values()),
        examples=_examples(sorted(dupes)),
    )


def tasks_spanning_splits(rows: Sequence[Mapping[str, Any]]) -> IntegrityIssue | None:
    """One task appearing in two splits.

    Task-level splitting exists to prevent exactly this. When it happens, the
    test set contains rows whose siblings trained the model, and the reported
    number is optimistic by an amount nobody can estimate after the fact.
    """
    splits_by_task: dict[str, set[str]] = {}
    for row in rows:
        task = str(row.get("task_id"))
        splits_by_task.setdefault(task, set()).add(str(row.get("split") or ""))
    offenders = {t: s for t, s in splits_by_task.items() if len(s) > 1}
    if not offenders:
        return None
    return IntegrityIssue(
        check="task_spans_splits",
        detail="a task_id appears in more than one split, so the evaluation "
               "set contains siblings of training rows",
        n_rows=len(offenders),
        examples=_examples(f"{t} in {sorted(s)}" for t, s in
                           sorted(offenders.items())),
    )


def orphan_ladder_steps(rows: Sequence[Mapping[str, Any]]) -> IntegrityIssue | None:
    """A repair row whose parent is absent, or absent from this store.

    Breaks sequential replay: the ladder cannot be reconstructed, so any
    cascade or repair policy computed over it is measuring something other than
    the policy it claims to measure.
    """
    known = {str(row.get("rollout_id")) for row in rows}
    orphans = []
    for row in rows:
        if int(row.get("ladder_step") or 0) <= 0:
            continue
        parent = row.get("parent_rollout_id")
        if not parent or str(parent) not in known:
            orphans.append(str(row.get("rollout_id")))
    if not orphans:
        return None
    return IntegrityIssue(
        check="orphan_ladder_step",
        detail="a row at ladder_step > 0 has no parent in this store, so the "
               "ladder it belongs to cannot be replayed",
        n_rows=len(orphans),
        examples=_examples(orphans),
    )


def impossible_counts(rows: Sequence[Mapping[str, Any]]) -> IntegrityIssue | None:
    """Counts that cannot be true, which must be rejected rather than clamped.

    Clamping `passed > total` to `total` turns a corrupt row into a perfect
    score — the single most flattering direction the error could be resolved in.
    """
    offenders: list[str] = []
    for row in rows:
        rollout_id = str(row.get("rollout_id"))
        for passed_key, total_key in (("visible_passed", "visible_total"),
                                      ("hidden_passed", "hidden_total")):
            passed, total = row.get(passed_key), row.get(total_key)
            if passed is None or total is None:
                continue
            if int(passed) < 0 or int(total) < 0 or int(passed) > int(total):
                offenders.append(f"{rollout_id}:{passed_key}={passed}/{total}")
        for key in ("prefill_tokens", "decode_tokens", "gpu_seconds",
                    "imputed_latency_s", "grade_duration_s"):
            value = row.get(key)
            if value is not None and float(value) < 0:
                offenders.append(f"{rollout_id}:{key}={value}")
    if not offenders:
        return None
    return IntegrityIssue(
        check="impossible_counts",
        detail="a count is negative or exceeds its total; clamping one would "
               "turn a corrupt row into a perfect score",
        n_rows=len(offenders),
        examples=_examples(offenders),
    )


def control_characters(rows: Sequence[Mapping[str, Any]]) -> IntegrityIssue | None:
    """A control character in a field that keys a join.

    Hostile text in `code` or `error_class` is content and must survive intact.
    A null byte in a `task_id` is not content — it silently breaks the join
    key, the split manifest, and any path derived from it, while most of the
    string still renders normally.
    """
    offenders = []
    for row in rows:
        for field in _IDENTITY_FIELDS:
            if _has_control_chars(row.get(field)):
                offenders.append(f"{field}={row.get(field)!r}")
    if not offenders:
        return None
    return IntegrityIssue(
        check="control_characters",
        detail="an identity field contains a control character, which breaks "
               "the join key it is used as while still rendering as text",
        n_rows=len(offenders),
        examples=_examples(offenders),
    )


def batched_serving_rows(rows: Sequence[Mapping[str, Any]]) -> IntegrityIssue | None:
    """A latency measurement taken under batching.

    The exact category error the `mode`/`batch_size` pair exists to prevent.
    Under concurrency, elapsed time is a queue-depth measurement wearing the
    label of a latency measurement, and it looks entirely plausible.
    """
    offenders = [
        str(row.get("rollout_id")) for row in rows
        if str(row.get("mode")) == "serving" and int(row.get("batch_size") or 1) > 1
    ]
    if not offenders:
        return None
    return IntegrityIssue(
        check="batched_serving_row",
        detail="a serving-mode row was produced under batching, so its "
               "wall-clock measures queue depth rather than the model",
        n_rows=len(offenders),
        examples=_examples(offenders),
    )


def single_arm(rows: Sequence[Mapping[str, Any]]) -> IntegrityIssue | None:
    """Only one arm present.

    Not a degenerate case to tolerate. The whole question is which arm to
    choose; with one arm there is no choice, and a paired comparison over it
    has no discordant pairs and reports a p-value of 1.0 rather than an error.
    """
    arms = sorted({str(row.get("arm")) for row in rows})
    if len(arms) > 1:
        return None
    return IntegrityIssue(
        check="single_arm",
        detail=f"only one arm is present ({arms}); a routing policy needs at "
               f"least two, and a paired comparison over one arm is a "
               f"comparison of a thing with itself",
        n_rows=len(rows),
    )


#: Run in this order so the most structural problem is reported first. A store
#: with duplicate cells will usually also trip later checks, and the duplicate
#: is the one to fix.
CHECKS = (
    duplicate_rollout_ids,
    tasks_spanning_splits,
    orphan_ladder_steps,
    impossible_counts,
    control_characters,
    batched_serving_rows,
    single_arm,
)


def check_integrity(rows: Sequence[Mapping[str, Any]], *,
                    skip: Iterable[str] = ()) -> list[IntegrityIssue]:
    """Every integrity problem in `rows`, not just the first.

    `skip` names checks to omit by slug. It exists for the one legitimate case
    — inspecting a store you already know is broken, to see what else is wrong
    with it — and not as a way to make a training run start.
    """
    skipped = {str(s) for s in skip}
    issues = []
    for check in CHECKS:
        if check.__name__ in skipped:
            continue
        issue = check(rows)
        if issue is not None and issue.check not in skipped:
            issues.append(issue)
    return issues


def format_issues(issues: Sequence[IntegrityIssue], *, run_id: str) -> str:
    """One message describing everything wrong, rather than the first thing."""
    lines = [
        f"run {run_id} failed {len(issues)} integrity "
        f"{'check' if len(issues) == 1 else 'checks'}:",
    ]
    lines.extend(f"  {issue}" for issue in issues)
    lines.append(
        "  Each of these produces a plausible number rather than an error, "
        "which is why they are refused here rather than survived."
    )
    return "\n".join(lines)
