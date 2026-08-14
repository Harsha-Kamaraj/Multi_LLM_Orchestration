"""Why generations fail — auto-tagged from what the grader actually recorded.

Accuracy says how often the system is wrong. This says *how*, which is the part
that changes what anyone does next. A 20-point arm gap made of timeouts is a
serving problem; the same gap made of wrong-answer failures is a capability
problem; made of reward hacks it is a measurement problem and the gap is not
real at all.

Every category here is derived from a field R2's grader already emits —
`error_class`, `hack_flags`, `finish_reason`, and the visible/hidden pair. None
of it is inferred from text, because a taxonomy built on string matching over
tracebacks drifts the moment a dependency changes its error message.

The categories are ordered by severity, and each row gets exactly one: the most
severe that applies. Overlapping categories would double-count, and a taxonomy
whose columns sum to more than 100% teaches readers to distrust it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .loading import Rollouts

# Most severe first. A row is tagged with the first that matches.
#
# `reward_hack` outranks everything, including success: a generation that passed
# by defeating the harness is not a success with a caveat, it is a measurement
# failure, and burying it under `solved` is how a hacked run gets reported as a
# good one.
CATEGORIES: tuple[str, ...] = (
    "reward_hack",
    "harness_error",
    "timeout",
    "empty_code",
    "syntax_error",
    "truncated",
    "runtime_error",
    "wrong_answer",
    "overfit_visible",
    "solved",
)

# What each category tells you to do about it. Carried in the report so a reader
# does not have to reconstruct the reasoning from a bare count.
MEANING: dict[str, str] = {
    "reward_hack": "passed by defeating the harness — the measurement is wrong, not the model",
    "harness_error": "the grader failed, not the code — infrastructure, exclude from capability claims",
    "timeout": "did not finish in the limit — may be an infinite loop or a genuinely slow solution",
    "empty_code": "no code extracted — a prompt or extraction problem, not a capability one",
    "syntax_error": "output did not parse — the model produced prose, or extraction took the wrong block",
    "truncated": "hit the token limit — a failed generation that grades as a capability gap",
    "runtime_error": "raised during execution — a real failure, and the most informative one",
    "wrong_answer": "ran cleanly and returned the wrong result — the honest capability failure",
    "overfit_visible": "passed visible tests, failed hidden ones — the failure a cascade cannot catch",
    "solved": "all hidden tests passed",
}

_HACK_FLAGS: frozenset[str] = frozenset({
    "hardcoded_visible_case", "bare_except_pass", "reads_test_file",
    "sys_exit_or_skip", "test_file_deleted", "test_file_modified",
})


@dataclass(frozen=True)
class Taxonomy:
    """Per-category counts, overall and per arm."""

    counts: dict[str, int]
    by_arm: dict[str, dict[str, int]]
    n: int

    def rate(self, category: str) -> float:
        return self.counts.get(category, 0) / self.n if self.n else float("nan")

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.rate("solved")

    def dominant_failure(self) -> str | None:
        """The largest non-success category. What to work on next."""
        failures = {k: v for k, v in self.counts.items()
                    if k != "solved" and v > 0}
        return max(failures, key=lambda k: failures[k]) if failures else None

    def arm_delta(self, category: str, a: str, b: str) -> float:
        """Difference in a category's rate between two arms.

        This is the number that turns "the large arm is 19 points better" into
        a statement about *why*.
        """
        na = sum(self.by_arm.get(a, {}).values()) or 1
        nb = sum(self.by_arm.get(b, {}).values()) or 1
        return self.by_arm.get(a, {}).get(category, 0) / na - \
            self.by_arm.get(b, {}).get(category, 0) / nb

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "counts": dict(sorted(self.counts.items())),
            "rates": {k: self.rate(k) for k in sorted(self.counts)},
            "by_arm": {a: dict(sorted(c.items()))
                       for a, c in sorted(self.by_arm.items())},
            "dominant_failure": self.dominant_failure(),
            "meaning": MEANING,
        }

    def __str__(self) -> str:
        arms = sorted(self.by_arm)
        head = f"{'category':<18}{'n':>7}{'rate':>8}" + \
            "".join(f"{a:>10}" for a in arms)
        lines = [head]
        for name in CATEGORIES:
            if not self.counts.get(name):
                continue
            row = f"{name:<18}{self.counts[name]:>7}{self.rate(name):>8.3f}"
            for arm in arms:
                total = sum(self.by_arm[arm].values()) or 1
                row += f"{self.by_arm[arm].get(name, 0) / total:>10.3f}"
            lines.append(row)
        return "\n".join(lines)


def classify_row(
    *,
    solved: float,
    visible_frac: float,
    error_class: str | None,
    finish_reason: str | None,
    hack_flags: Sequence[str] | None,
) -> str:
    """Assign one row to exactly one category, most severe first."""
    if hack_flags and set(hack_flags) & _HACK_FLAGS:
        return "reward_hack"
    if error_class == "harness_error":
        return "harness_error"
    if error_class == "timeout":
        return "timeout"
    if error_class == "empty_code":
        return "empty_code"
    if error_class == "syntax_error":
        return "syntax_error"

    if solved >= 1.0:
        return "solved"

    # Truncation is checked *after* success: a generation that hit the token
    # limit and still passed every hidden test is solved, and calling it a
    # failure would understate the arm.
    if finish_reason == "length":
        return "truncated"
    if error_class == "runtime_error":
        return "runtime_error"

    # Passed everything it could see and still failed. The failure a cascade
    # structurally cannot catch, because the cascade's escalation signal is
    # exactly the visible tests this row satisfied.
    if np.isfinite(visible_frac) and visible_frac >= 1.0:
        return "overfit_visible"
    return "wrong_answer"


def classify(rows: Iterable[Mapping[str, object]]) -> Taxonomy:
    """Tag raw rollout rows. Works before the loader, for triage on a broken run."""
    counts = dict.fromkeys(CATEGORIES, 0)
    by_arm: dict[str, dict[str, int]] = {}
    n = 0

    for row in rows:
        n += 1
        hidden_total = row.get("hidden_total")
        hidden_passed = row.get("hidden_passed")
        solved = float(
            hidden_total not in (None, 0) and hidden_passed == hidden_total
        )
        visible_total = row.get("visible_total") or 0
        visible_frac = (
            float(row.get("visible_passed") or 0) / float(visible_total)
            if visible_total else float("nan")
        )
        category = classify_row(
            solved=solved,
            visible_frac=visible_frac,
            error_class=row.get("error_class"),  # type: ignore[arg-type]
            finish_reason=row.get("finish_reason"),  # type: ignore[arg-type]
            hack_flags=row.get("hack_flags"),  # type: ignore[arg-type]
        )
        counts[category] += 1
        arm = str(row.get("arm", "?"))
        by_arm.setdefault(arm, dict.fromkeys(CATEGORIES, 0))[category] += 1

    return Taxonomy(counts=counts, by_arm=by_arm, n=n)


def classify_store(store: Rollouts) -> Taxonomy:
    """Tag a loaded store, reusing the columns the loader already derived."""
    counts = dict.fromkeys(CATEGORIES, 0)
    by_arm: dict[str, dict[str, int]] = {}

    solved = store["_solved"]
    visible = store["_visible_frac"]
    arms = store["arm"]
    truncated = store["truncated"]
    hacked = store["hacked"]

    for i in range(len(store)):
        category = classify_row(
            solved=0.0 if np.isnan(solved[i]) else float(solved[i]),
            visible_frac=float(visible[i]),
            error_class=None,
            finish_reason="length" if truncated[i] else None,
            hack_flags=("hardcoded_visible_case",) if hacked[i] else (),
        )
        counts[category] += 1
        by_arm.setdefault(str(arms[i]), dict.fromkeys(CATEGORIES, 0))[category] += 1

    return Taxonomy(counts=counts, by_arm=by_arm, n=len(store))


def explain_gap(taxonomy: Taxonomy, *, small: str = "small", large: str = "large") -> list[str]:
    """Attribute an arm gap to categories, largest contribution first.

    Turns "the large arm is 19 points better" into a claim about mechanism —
    which is the difference between a benchmark number and a finding.
    """
    if small not in taxonomy.by_arm or large not in taxonomy.by_arm:
        return []
    deltas = sorted(
        ((c, taxonomy.arm_delta(c, large, small)) for c in CATEGORIES),
        key=lambda kv: abs(kv[1]), reverse=True,
    )
    out = []
    for category, delta in deltas:
        if abs(delta) < 0.005:
            continue
        direction = "more" if delta > 0 else "fewer"
        out.append(
            f"{large} has {abs(delta):.1%} {direction} {category} "
            f"— {MEANING[category]}"
        )
    return out
