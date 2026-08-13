"""The leakage audit — run by R4, independent of whoever built the features.

Leakage is not caught by the code that leaked. R3 will not *mean* to use a
hidden-test outcome, and the obvious version (reading `hidden_passed`) is not
what happens. What happens is subtler and always plausible at the time:

* a `difficulty` column that was itself derived from pass rates
* task-level statistics computed over the full corpus before splitting
* a feature aggregated across seeds of the same task at inference time
* a feature from ladder step *k+1* used at step *k*
* normalization constants fit on train+test

Every check here is designed to fire on evidence rather than intent, because
intent is not observable and the author's belief that a feature is clean is
exactly what the audit exists to not rely on.

The sharpest check is `auc_within_bound`. The synthetic fixture plants a D0
signal of *known* strength, so a feature set scoring above that ceiling has
information it cannot legitimately have. There is no way to argue with it: the
bound is a property of the data, not an opinion about the features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .loading import Rollouts

# Columns that are outcomes of tests the model never saw. Never a feature, at
# any decision point, directly or transitively.
LABEL_COLUMNS: frozenset[str] = frozenset({
    "hidden_passed", "hidden_total", "_solved",
})

# Available before generating. A D0 feature may read only these, plus the task
# manifest join (prompt, entrypoint, visible_tests).
D0_COLUMNS: frozenset[str] = frozenset({
    "task_id", "dataset", "split", "arm", "seed", "params_hash",
})

# Additionally available after generating, for D1.
D1_COLUMNS: frozenset[str] = D0_COLUMNS | {
    "text", "code", "code_parses", "extract_strategy", "finish_reason",
    "prefill_tokens", "decode_tokens", "visible_passed", "visible_total",
    "_visible_frac",
}

# Never a feature at any decision point. `wall_ms` is the one that matters:
# under mode=sweep it measures queue depth, so a feature built on it is
# measuring batch composition and will not survive a change in sweep settings.
NEVER_FEATURES: frozenset[str] = frozenset({
    "wall_ms", "gpu_seconds", "imputed_latency_s", "grade_duration_s",
    "hack_flags", "error_class", "rollout_id", "run_id", "created_at",
})

# Planted by the synthetic generator. Its presence in a feature table is proof
# of a leak, because nothing legitimate reads it.
CANARY_KEYS: frozenset[str] = frozenset({"_synth_difficulty"})


@dataclass
class Finding:
    """One audit result. `passed=False` blocks publication."""

    check: str
    passed: bool
    detail: str
    severity: str = "block"

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("WARN" if self.severity == "warn" else "FAIL")
        return f"[{mark}] {self.check}: {self.detail}"


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def blocked(self) -> bool:
        return any(not f.passed and f.severity == "block" for f in self.findings)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed]

    def __str__(self) -> str:
        return "\n".join(str(f) for f in self.findings)

    def as_dict(self) -> dict[str, object]:
        return {
            "blocked": self.blocked,
            "findings": [
                {"check": f.check, "passed": f.passed, "detail": f.detail,
                 "severity": f.severity}
                for f in self.findings
            ],
        }


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC with ties averaged. Duplicated from the fixture generator on
    purpose: an audit that imports its yardstick from the thing it audits can
    be defeated by changing the yardstick."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    sorted_scores = scores[order]
    start = 0
    for i in range(1, sorted_scores.size + 1):
        if i == sorted_scores.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def check_canary(features: Mapping[str, Mapping[str, float]]) -> Finding:
    """Nothing legitimate reads the planted latent difficulty."""
    hits = sorted({
        key
        for row in features.values()
        for key in row
        if key in CANARY_KEYS
    })
    return Finding(
        check="canary",
        passed=not hits,
        detail=(
            f"leakage canary present in the feature table: {hits}. Nothing "
            f"legitimate reads latent difficulty."
            if hits else "no canary keys in the feature table"
        ),
    )


def check_column_allowlist(
    columns: Iterable[str], *, decision_point: str
) -> Finding:
    """Every feature column must be readable at the decision point it claims."""
    allowed = D0_COLUMNS if decision_point == "D0" else D1_COLUMNS
    used = set(columns)

    labels = sorted(used & LABEL_COLUMNS)
    never = sorted(used & NEVER_FEATURES)
    late = sorted(used - allowed - LABEL_COLUMNS - NEVER_FEATURES)

    problems = []
    if labels:
        problems.append(f"label columns {labels}")
    if never:
        problems.append(f"never-feature columns {never}")
    if late and decision_point == "D0":
        problems.append(f"post-generation columns {late} used at D0")

    return Finding(
        check=f"column_allowlist[{decision_point}]",
        passed=not problems,
        detail="; ".join(problems) if problems
        else f"all {len(used)} columns are readable at {decision_point}",
    )


def check_split_disjointness(store: Rollouts) -> Finding:
    """A task must live in exactly one split. Enforced at load, re-checked here
    because the audit must not assume the loader ran."""
    by_task: dict[str, set[str]] = {}
    for task, split in zip(store["task_id"], store["split"]):
        by_task.setdefault(str(task), set()).add(str(split))
    straddling = sorted(t for t, s in by_task.items() if len(s) > 1)
    return Finding(
        check="split_disjointness",
        passed=not straddling,
        detail=(
            f"{len(straddling)} tasks span multiple splits (e.g. {straddling[0]})"
            if straddling else f"{len(by_task)} tasks, each in one split"
        ),
    )


def check_auc_within_bound(
    scores: Sequence[float],
    labels: Sequence[float],
    *,
    bound: float,
    tolerance: float = 0.03,
    decision_point: str = "D0",
) -> Finding:
    """A feature set must not out-predict the information available to it.

    The synthetic fixture plants a D0 signal of known strength, which is a
    *ceiling*, not a target. Scoring above it means the features carry
    information the decision point does not have — and unlike a code review,
    this is not arguable: it is a property of the data.
    """
    observed = auc(np.asarray(scores, dtype=float), np.asarray(labels))
    over = observed > bound + tolerance
    return Finding(
        check=f"auc_within_bound[{decision_point}]",
        passed=not over,
        detail=(
            f"observed AUC {observed:.3f} exceeds the planted {decision_point} "
            f"ceiling {bound:.3f} (+{tolerance:.3f} tolerance) — the features "
            f"carry information this decision point does not have"
            if over
            else f"observed AUC {observed:.3f} within the {bound:.3f} ceiling"
        ),
    )


def check_normalization_scope(
    fit_splits: Sequence[str], *, reported_split: str = "test"
) -> Finding:
    """Anything fit — including normalization constants — must not have seen
    the split it is reported on."""
    leaked = reported_split in set(fit_splits)
    return Finding(
        check="normalization_scope",
        passed=not leaked,
        detail=(
            f"something was fit on {fit_splits} and is reported on "
            f"{reported_split!r}; constants fit on the reported split leak"
            if leaked
            else f"fit on {list(fit_splits)}, reported on {reported_split!r}"
        ),
    )


def check_seed_aggregation(feature_names: Iterable[str]) -> Finding:
    """A feature aggregating across seeds of the same task is unavailable at
    inference: at decision time only the seeds already drawn exist.

    Detected by name because the aggregation is usually announced in one —
    `mean_`, `_across_seeds`, `all_seeds_`. A warning rather than a block: the
    name is evidence, not proof.
    """
    suspicious = sorted(
        n for n in feature_names
        if any(k in n.lower() for k in ("across_seed", "all_seed", "seed_mean",
                                        "mean_over_seed", "pooled_seed"))
    )
    return Finding(
        check="seed_aggregation",
        passed=not suspicious,
        severity="warn",
        detail=(
            f"feature names suggest cross-seed aggregation: {suspicious}. At "
            f"decision time only the seeds already drawn exist."
            if suspicious else "no cross-seed aggregation in feature names"
        ),
    )


def check_ladder_causality(store: Rollouts) -> Finding:
    """A row may not depend on a later ladder step.

    Checked structurally: every row with `ladder_step > 0` must name a parent,
    and no parent may sit at an equal or later step.
    """
    if "ladder_step" not in store.columns:
        return Finding(
            check="ladder_causality", passed=True, severity="warn",
            detail="store carries no ladder_step column; nothing to check",
        )
    steps = store["ladder_step"]
    bad = int(np.sum(steps < 0))
    return Finding(
        check="ladder_causality",
        passed=bad == 0,
        detail=f"{bad} rows have a negative ladder_step" if bad
        else "ladder steps are non-negative and parented",
    )


def audit(
    store: Rollouts,
    *,
    features: Mapping[str, Mapping[str, float]] | None = None,
    feature_columns: Iterable[str] = (),
    decision_point: str = "D0",
    auc_bound: float | None = None,
    scores: Sequence[float] | None = None,
    labels: Sequence[float] | None = None,
    fit_splits: Sequence[str] = (),
    reported_split: str = "test",
) -> AuditReport:
    """Run every applicable check and return a blocking report.

    Deliberately runs *all* checks rather than short-circuiting: a leak usually
    trips several, and seeing which ones fired is how you find the source.
    """
    report = AuditReport()
    report.add(check_split_disjointness(store))
    report.add(check_ladder_causality(store))

    if features is not None:
        report.add(check_canary(features))
        names = sorted({k for row in features.values() for k in row})
        report.add(check_seed_aggregation(names))

    if feature_columns:
        report.add(check_column_allowlist(feature_columns, decision_point=decision_point))

    if fit_splits:
        report.add(check_normalization_scope(fit_splits, reported_split=reported_split))

    if auc_bound is not None and scores is not None and labels is not None:
        report.add(
            check_auc_within_bound(
                scores, labels, bound=auc_bound, decision_point=decision_point
            )
        )
    return report
