"""The router confusion matrix — why a router won or lost, not just whether.

Accuracy alone cannot distinguish a router that escalated wisely from one that
escalated constantly, nor a router that lost accuracy from one facing tasks no
arm can solve. Five outcomes separate those cases:

    correct_small        routed cheap, solved            the win condition
    false_escalation     paid for large, small sufficed  wasted money
    missed_escalation    stayed small, large would win   lost accuracy
    correct_escalation   escalated, large solved it      money well spent
    unsolvable           neither arm solves it           not a routing failure

`unsolvable` is the one people forget, and it matters most. It is the ceiling: a
router cannot be blamed for tasks no arm solves, and folding them into the error
rate makes every router look worse than it is — uniformly, which means it also
compresses the differences between routers.

Computing this requires the hidden outcome for *both* arms, so it is an
analysis-time artifact only. Nothing here may be fed back into a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loading import Rollouts
from .policies import Outcome

OUTCOMES = (
    "correct_small",
    "false_escalation",
    "missed_escalation",
    "correct_escalation",
    "unsolvable",
)


@dataclass(frozen=True)
class Confusion:
    """Counts and rates over the five routing outcomes."""

    counts: dict[str, int]
    n: int
    escalation_rate: float
    warnings: list[str] = field(default_factory=list)

    def rate(self, outcome: str) -> float:
        return self.counts[outcome] / self.n if self.n else float("nan")

    @property
    def routable(self) -> int:
        """Tasks where the routing decision could have mattered.

        Excludes `unsolvable`. Denominators built on this are the honest ones
        for judging a router.
        """
        return self.n - self.counts["unsolvable"]

    @property
    def regret(self) -> float:
        """Fraction of *routable* tasks the router got wrong in either direction.

        One number that penalizes over- and under-escalation symmetrically,
        which accuracy does not: a router that escalates everything loses no
        accuracy at all and is still bad.
        """
        if not self.routable:
            return float("nan")
        wrong = self.counts["false_escalation"] + self.counts["missed_escalation"]
        return wrong / self.routable

    def as_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = dict(self.counts)
        out.update({
            "n": self.n, "routable": self.routable,
            "escalation_rate": self.escalation_rate, "regret": self.regret,
        })
        for name in OUTCOMES:
            out[f"{name}_rate"] = self.rate(name)
        return out

    def __str__(self) -> str:
        lines = [f"{'outcome':<20}{'count':>8}{'rate':>9}"]
        for name in OUTCOMES:
            lines.append(f"{name:<20}{self.counts[name]:>8}{self.rate(name):>9.3f}")
        lines.append(f"{'regret (routable)':<20}{'':>8}{self.regret:>9.3f}")
        return "\n".join(lines)


def confusion(
    outcome: Outcome,
    store: Rollouts,
    *,
    small: str = "small",
    large: str = "large",
) -> Confusion:
    """Classify each task by what the router did and what was achievable.

    Achievability is judged per task by whether *any* seed of an arm solved it.
    Using a single seed would make the classification depend on which seed
    happened to be drawn, and a task's label would flicker between
    `missed_escalation` and `unsolvable` across otherwise identical runs.
    """
    solved = store.arm_matrix("_solved")
    for arm in (small, large):
        if arm not in solved:
            raise KeyError(f"arm {arm!r} not in store; have {sorted(solved)}")

    small_can = np.nanmax(solved[small], axis=1) >= 1.0
    large_can = np.nanmax(solved[large], axis=1) >= 1.0

    # A policy escalated on a task if it chose the large arm for any replicate.
    # `->` covers the cascade's composite label.
    actions = outcome.action.astype(str)
    escalated = np.array(
        [any(large in a for a in row) for row in actions], dtype=bool
    )

    counts = dict.fromkeys(OUTCOMES, 0)
    for i in range(len(escalated)):
        if not small_can[i] and not large_can[i]:
            counts["unsolvable"] += 1
        elif escalated[i]:
            counts["correct_escalation" if not small_can[i] else "false_escalation"] += 1
        else:
            counts["correct_small" if small_can[i] else "missed_escalation"] += 1

    n = int(len(escalated))
    result = Confusion(
        counts=counts, n=n,
        escalation_rate=float(escalated.mean()) if n else float("nan"),
    )

    warnings: list[str] = []
    if result.rate("unsolvable") > 0.30:
        warnings.append(
            f"{result.rate('unsolvable'):.1%} of tasks are unsolvable by either "
            f"arm. The routable headroom is small; report accuracy over routable "
            f"tasks alongside the raw number or the comparison looks flatter "
            f"than it is."
        )

    # A policy with one distinct action is a *fixed* policy, not a degenerate
    # router. Warning that `always_small` behaves like `always_small` is noise,
    # and noise in a warning channel trains people to ignore the channel — which
    # costs the one time it fires on something real.
    is_router = len({a for row in actions for a in row}) > 1
    if is_router:
        if result.escalation_rate > 0.95:
            warnings.append(
                "escalates on >95% of tasks — effectively always_large, and its "
                "accuracy should be read as such rather than as routing."
            )
        if result.escalation_rate < 0.05:
            warnings.append(
                "escalates on <5% of tasks — effectively always_small."
            )
    return Confusion(
        counts=counts, n=n, escalation_rate=result.escalation_rate,
        warnings=warnings,
    )


def oracle_headroom(store: Rollouts, *, small: str = "small", large: str = "large") -> dict[str, float]:
    """How much routing could possibly buy on this corpus.

    Reported before any policy is compared. If `escalation_helps` is tiny, no
    router can win by much and a small measured gap is not a weak policy — it
    is a saturated problem. Conflating the two is how a null result gets blamed
    on the wrong thing.
    """
    solved = store.arm_matrix("_solved")
    small_can = np.nanmax(solved[small], axis=1) >= 1.0
    large_can = np.nanmax(solved[large], axis=1) >= 1.0
    n = len(small_can)
    return {
        "n_tasks": float(n),
        "small_only": float(np.mean(small_can & ~large_can)),
        "large_only": float(np.mean(~small_can & large_can)),
        "both": float(np.mean(small_can & large_can)),
        "neither": float(np.mean(~small_can & ~large_can)),
        # The tasks a router exists to catch.
        "escalation_helps": float(np.mean(~small_can & large_can)),
        # Where a router can save money without losing accuracy.
        "escalation_wasteful": float(np.mean(small_can)),
    }
