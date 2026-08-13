"""The cost–accuracy frontier, and comparison *at matched cost*.

The single most common way a routing result is oversold is comparing at
different price points. A router that is more accurate than `always_small` and
cheaper than `always_large` has demonstrated nothing on its own — that is what
interpolating between two arms does, and `random_route` does it for free.

So every comparison here is anchored to cost. `compare_at_matched_cost` finds
the accuracy each policy achieves *at the same spend* and reports the
difference. A win at a different price is not a win.

λ is swept at decision time, never baked into a model. One set of value heads
produces the entire curve, which is what makes the frontier a product dial
rather than a retraining job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .loading import Rollouts
from .policies import Outcome
from .stats import Interval, paired_diff_bootstrap


@dataclass(frozen=True)
class Point:
    """One policy evaluated at one λ."""

    policy: str
    lam: float
    accuracy: float
    cost: float
    latency_mean: float
    latency_p95: float

    @property
    def utility(self) -> float:
        return self.accuracy - self.lam * self.cost

    def as_dict(self) -> dict[str, float | str]:
        return {
            "policy": self.policy, "lam": self.lam, "accuracy": self.accuracy,
            "cost": self.cost, "latency_mean": self.latency_mean,
            "latency_p95": self.latency_p95, "utility": self.utility,
        }


def evaluate(outcome: Outcome, lam: float) -> Point:
    """Summarize one policy's outcome at one λ."""
    s = outcome.summary()
    return Point(
        policy=outcome.name, lam=lam, accuracy=s["accuracy"], cost=s["cost"],
        latency_mean=s["latency_mean"], latency_p95=s["latency_p95"],
    )


def sweep(
    policies: Mapping[str, Callable[[Rollouts], Outcome]],
    store: Rollouts,
    lams: Sequence[float],
) -> dict[str, list[Point]]:
    """Evaluate every policy at every λ.

    Fixed policies are λ-independent — `always_small` costs what it costs — so
    their curve is flat in (cost, accuracy) and varies only in utility. That is
    correct and worth seeing: it shows exactly which region of the frontier a
    fixed arm happens to be optimal in.
    """
    out: dict[str, list[Point]] = {}
    for name, fn in policies.items():
        outcome = fn(store)
        out[name] = [evaluate(outcome, lam) for lam in lams]
    return out


def pareto_front(points: Sequence[Point]) -> list[Point]:
    """Points not dominated on (cost ↓, accuracy ↑).

    Ties broken toward lower cost: two policies at identical accuracy are not
    equivalent, and reporting the more expensive one as frontier-optimal would
    flatter whichever policy happens to spend more.
    """
    ordered = sorted(points, key=lambda p: (p.cost, -p.accuracy))
    front: list[Point] = []
    best_accuracy = -np.inf
    for point in ordered:
        if point.accuracy > best_accuracy + 1e-12:
            front.append(point)
            best_accuracy = point.accuracy
    return front


def accuracy_at_cost(points: Sequence[Point], target_cost: float) -> float | None:
    """Linearly interpolate a policy's accuracy at `target_cost`.

    Returns `None` outside the policy's achievable cost range rather than
    extrapolating. Extrapolating a frontier past its measured endpoints invents
    a capability the policy was never shown to have — and it is exactly the
    region where a losing policy would like to be compared.
    """
    usable = sorted({(p.cost, p.accuracy) for p in points})
    if len(usable) < 2:
        return usable[0][1] if usable and abs(usable[0][0] - target_cost) < 1e-9 else None

    costs = np.array([c for c, _ in usable])
    accs = np.array([a for _, a in usable])
    if target_cost < costs[0] or target_cost > costs[-1]:
        return None
    return float(np.interp(target_cost, costs, accs))


@dataclass(frozen=True)
class MatchedComparison:
    """Two policies compared at the same spend."""

    a: str
    b: str
    cost: float
    accuracy_a: float
    accuracy_b: float

    @property
    def difference(self) -> float:
        return self.accuracy_a - self.accuracy_b

    def __str__(self) -> str:
        return (
            f"{self.a} vs {self.b} at cost={self.cost:.4g}: "
            f"{self.accuracy_a:.4f} - {self.accuracy_b:.4f} = {self.difference:+.4f}"
        )


def compare_at_matched_cost(
    a: Sequence[Point], b: Sequence[Point], *, n_grid: int = 25
) -> list[MatchedComparison]:
    """Compare two policies across their overlapping cost range.

    Only the overlap is reported. Outside it one policy has no measurement, and
    comparing against an extrapolation is how "we beat the cascade" survives a
    result where the two were never priced the same.
    """
    a_costs = [p.cost for p in a]
    b_costs = [p.cost for p in b]
    low = max(min(a_costs), min(b_costs))
    high = min(max(a_costs), max(b_costs))
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        return []

    name_a = a[0].policy if a else "a"
    name_b = b[0].policy if b else "b"
    grid = [low] if high == low else list(np.linspace(low, high, n_grid))

    out: list[MatchedComparison] = []
    for cost in grid:
        acc_a = accuracy_at_cost(a, float(cost))
        acc_b = accuracy_at_cost(b, float(cost))
        if acc_a is None or acc_b is None:
            continue
        out.append(
            MatchedComparison(name_a, name_b, float(cost), acc_a, acc_b)
        )
    return out


def paired_accuracy_difference(
    a: Outcome, b: Outcome, *, n_resamples: int = 10_000, seed: int = 0
) -> Interval:
    """Interval on `accuracy(a) - accuracy(b)`, paired on tasks.

    Policies with different replicate counts — `best_of_n_small` consumes every
    seed and yields one — are reduced to per-task means first, which keeps the
    comparison paired at the level the bootstrap clusters on anyway.
    """
    if a.solved.shape != b.solved.shape:
        a, b = a.per_task(), b.per_task()
    return paired_diff_bootstrap(
        a.solved, b.solved, n_resamples=n_resamples, seed=seed
    )


def frontier_dominates(
    challenger: Sequence[Point], incumbent: Sequence[Point], *, tolerance: float = 0.0
) -> bool:
    """Whether `challenger` is at least as accurate at every matched cost.

    Deliberately strict. The project's claim is not "the policy wins somewhere",
    which any curve crossing another achieves by accident — it is that the
    policy wins in a region, with an interval excluding zero. This function
    answers the stronger question, and it is expected to return False often.
    """
    matched = compare_at_matched_cost(challenger, incumbent)
    return bool(matched) and all(m.difference >= -tolerance for m in matched)
