"""The Phase 0 gate — four quantities, adjudicated by code.

ROADMAP.md states the gate as a table. A table is a promise; this is the thing
that keeps it. R4 adjudicates because R4 is the role with no stake in passing:
the engineer who builds the policy should not be the one who decides whether the
premise survived.

The four quantities and what failing each one means:

| Quantity            | Threshold | If it fails                                    |
|---------------------|-----------|------------------------------------------------|
| `A_large - A_small` |   ≥ 8 pp  | Arms are not differentiated — shrink the small  |
| `A_oracle - A_large`|   ≥ 5 pp  | The large arm dominates; only cost is winnable  |
| `AUC_D0`            |   ≥ 0.65  | Pre-generation routing is dead. Move to D1      |
| `AUC_D1`            |   ≥ 0.75  | **Hard stop.** Neither point has signal         |

Only `AUC_D1` is a hard stop. The others change the plan rather than ending it,
and saying so in advance is what stops a disappointing number being renegotiated
into a passing one after the fact.

`AUC_D0` needs prompt features, which are **not on the rollout row** — the row
carries `text` and `code`, model *output*. D0 features come from R2's task
manifest joined by `task_id`. When no corpus is supplied the D0 gate reports
`unmeasured` rather than guessing, because a gate that quietly skips itself is
worse than one that fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .heuristics import prompt_features
from .leakage import auc
from .loading import Rollouts
from .policies import always, oracle_router


@dataclass(frozen=True)
class Gate:
    """One gate quantity, measured against its threshold."""

    name: str
    value: float | None
    threshold: float
    hard_stop: bool
    on_failure: str
    detail: str = ""

    @property
    def measured(self) -> bool:
        return self.value is not None and np.isfinite(self.value)

    @property
    def passed(self) -> bool | None:
        """`None` when unmeasured — which is neither a pass nor a failure, and
        must not be collapsed into either."""
        return None if not self.measured else bool(self.value >= self.threshold)

    def __str__(self) -> str:
        if not self.measured:
            mark, shown = "····", "unmeasured"
        else:
            mark = "PASS" if self.passed else "FAIL"
            shown = f"{self.value:.4f}"
        stop = "  [HARD STOP]" if self.hard_stop and self.passed is False else ""
        return f"[{mark}] {self.name:<20} {shown:>12}  need ≥ {self.threshold:.4g}{stop}"


@dataclass
class GateReport:
    gates: list[Gate]

    @property
    def blocked(self) -> bool:
        """A failed hard stop ends the project. Nothing else does."""
        return any(g.hard_stop and g.passed is False for g in self.gates)

    @property
    def all_measured(self) -> bool:
        return all(g.measured for g in self.gates)

    @property
    def failures(self) -> list[Gate]:
        return [g for g in self.gates if g.passed is False]

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "all_measured": self.all_measured,
            "gates": [
                {"name": g.name, "value": g.value, "threshold": g.threshold,
                 "passed": g.passed, "hard_stop": g.hard_stop,
                 "on_failure": g.on_failure, "detail": g.detail}
                for g in self.gates
            ],
        }

    def __str__(self) -> str:
        lines = [str(g) for g in self.gates]
        for gate in self.failures:
            lines.append(f"       {gate.name}: {gate.on_failure}")
        if not self.all_measured:
            lines.append(
                "\nNot every quantity is measured. An unmeasured gate is not a "
                "passed gate — the phase does not advance on partial evidence."
            )
        return "\n".join(lines)


def _arm_accuracy(store: Rollouts, arm: str) -> float:
    return float(np.nanmean(store.arm_matrix("_solved")[arm]))


def evaluate(
    store: Rollouts,
    *,
    small: str = "small",
    large: str = "large",
    corpus: Sequence[Mapping[str, Any]] | None = None,
) -> GateReport:
    """Measure all four gate quantities from a graded store."""
    a_small = _arm_accuracy(store, small)
    a_large = _arm_accuracy(store, large)
    a_oracle = float(np.nanmean(oracle_router(small=small, large=large)(store).solved))

    gates = [
        Gate(
            name="A_large - A_small",
            value=a_large - a_small,
            threshold=0.08,
            hard_stop=False,
            on_failure="arms are not differentiated — shrink the small arm "
                       "before reaching for a bigger large arm, which costs TP=1",
            detail=f"{small}={a_small:.4f}  {large}={a_large:.4f}",
        ),
        Gate(
            name="A_oracle - A_large",
            value=a_oracle - a_large,
            threshold=0.05,
            hard_stop=False,
            on_failure="the large arm dominates; only cost savings are winnable, "
                       "not accuracy. Report the frontier, not a win",
            detail=f"oracle={a_oracle:.4f}",
        ),
        _auc_d0(store, corpus, small),
        _auc_d1(store, small),
    ]
    return GateReport(gates=gates)


def _auc_d1(store: Rollouts, small: str) -> Gate:
    """Can the visible-test outcome predict whether the hidden tests pass?

    This is the whole premise. If observing a candidate's visible-test result
    says nothing about whether it is actually correct, there is no escalation
    signal anywhere and the project has no mechanism left.
    """
    visible = store.arm_matrix("_visible_frac")[small].ravel()
    solved = store.arm_matrix("_solved")[small].ravel()
    keep = np.isfinite(visible) & np.isfinite(solved)
    value = auc(visible[keep], solved[keep]) if keep.any() else None
    return Gate(
        name="AUC_D1",
        value=value,
        threshold=0.75,
        hard_stop=True,
        on_failure="neither decision point carries signal. The premise is false "
                   "and the project stops here rather than continuing on hope",
        detail=f"n={int(keep.sum())} graded {small}-arm generations",
    )


def _auc_d0(
    store: Rollouts, corpus: Sequence[Mapping[str, Any]] | None, small: str
) -> Gate:
    """Can prompt-only features predict whether the small arm solves the task?

    Weak by expectation — 0.60–0.68 — and that weakness is the project's central
    finding rather than a disappointment. Reported honestly either way.
    """
    unmeasured = Gate(
        name="AUC_D0",
        value=None,
        threshold=0.65,
        hard_stop=False,
        on_failure="pre-generation routing is dead. Move to D1 — expected, "
                   "not fatal, and it is the finding rather than a setback",
        detail="no corpus supplied; prompt features are not on the rollout row, "
               "so this needs R2's task manifest joined by task_id",
    )
    if not corpus:
        return unmeasured

    features = {str(t["task_id"]): prompt_features(t) for t in corpus}
    tasks = store.ordered_tasks
    solved = store.arm_matrix("_solved")[small]
    # Any-seed success: a task the small arm solves on some seed is a task a
    # router should send to the small arm. Using a single seed would make the
    # label depend on which seed happened to be drawn.
    labels = (np.nanmax(solved, axis=1) >= 1.0).astype(float)

    best = None
    for name in sorted({k for row in features.values() for k in row}):
        scores = np.array(
            [features.get(t, {}).get(name, np.nan) for t in tasks], dtype=float
        )
        if not np.isfinite(scores).any():
            continue
        # A feature predicting failure is as informative as one predicting
        # success; the sign is a property of the feature, not of the signal.
        value = auc(scores, labels)
        if np.isfinite(value):
            value = max(value, 1.0 - value)
            best = value if best is None else max(best, value)

    if best is None:
        return unmeasured
    return Gate(
        name="AUC_D0",
        value=best,
        threshold=0.65,
        hard_stop=False,
        on_failure="pre-generation routing is dead. Move to D1 — expected, "
                   "not fatal, and it is the finding rather than a setback",
        detail=f"best single prompt feature over {len(tasks)} tasks",
    )
