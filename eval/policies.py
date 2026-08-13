"""The baseline family, replayed offline against a logged rollout store.

Every policy here answers the same question — for each task, which arm's logged
generation do we accept, and what did getting there cost? Because the store is
complete (every arm, every seed, every task), no importance weighting is
needed: the counterfactual was actually run.

Two things are easy to get wrong and are handled explicitly:

**Escalation is serial.** A cascade that runs small, observes failure, then runs
large has paid for *both*. Its latency is the sum, not the max. Charging only
the large arm would make the cascade look strictly better than it is, and it is
already the strongest baseline.

**Sampling is not free.** `best_of_n_small` spends n generations. Compared at
equal accuracy it must be compared at *its* cost, not the cost of one sample.
This is what stops "more samples helps" being mistaken for "routing helps".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .loading import Rollouts

# What a policy is allowed to see when it decides. The visible-test outcome is
# permitted; the hidden outcome never is, except for `oracle_router`, which is
# explicitly labelled as a ceiling rather than a competitor.
DECISION_COLUMNS = ("_visible_frac", "gpu_seconds", "imputed_latency_s")


@dataclass
class Outcome:
    """What a policy achieved, per task and replicate.

    Shape is `(n_tasks, n_replicates)`. Policies that consume every seed to make
    one decision — `best_of_n_small` — produce a single replicate, and
    `n_replicates == 1` is honest about that rather than tiling one value across
    three columns and pretending to three observations.
    """

    name: str
    solved: np.ndarray
    cost: np.ndarray
    latency: np.ndarray
    action: np.ndarray
    consumes_all_seeds: bool = False

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.solved, self.cost, self.latency)}
        if len(shapes) != 1:
            raise ValueError(f"{self.name}: outcome arrays disagree: {shapes}")

    @property
    def n_tasks(self) -> int:
        return self.solved.shape[0]

    def per_task(self) -> "Outcome":
        """Collapse replicates to a per-task mean.

        Used when comparing policies with different replicate counts. Averaging
        first keeps the comparison paired at the task level, which is the level
        the bootstrap clusters on anyway.
        """
        return Outcome(
            name=self.name,
            solved=self.solved.mean(axis=1, keepdims=True),
            cost=self.cost.mean(axis=1, keepdims=True),
            latency=self.latency.mean(axis=1, keepdims=True),
            action=self.action[:, :1],
            consumes_all_seeds=self.consumes_all_seeds,
        )

    def utility(self, lam: float) -> np.ndarray:
        """`accuracy - lam * cost`. The scalarization used only at decision and
        report time — never baked into a trained model."""
        return self.solved - lam * self.cost

    def summary(self) -> dict[str, float]:
        return {
            "accuracy": float(np.nanmean(self.solved)),
            "cost": float(np.nanmean(self.cost)),
            "latency_mean": float(np.nanmean(self.latency)),
            "latency_p95": float(np.nanpercentile(self.latency, 95)),
        }


def _grids(store: Rollouts) -> dict[str, dict[str, np.ndarray]]:
    """Per-arm `(n_tasks, n_seeds)` grids for everything a policy may read."""
    return {
        "solved": store.arm_matrix("_solved"),
        "visible": store.arm_matrix("_visible_frac"),
        "cost": store.arm_matrix("gpu_seconds"),
        "latency": store.arm_matrix("imputed_latency_s"),
    }


def always(arm: str) -> Callable[[Rollouts], Outcome]:
    """Always pull one arm. `always_small` is the cost floor, `always_large`
    the single-arm accuracy ceiling."""

    def policy(store: Rollouts) -> Outcome:
        g = _grids(store)
        if arm not in g["solved"]:
            raise KeyError(f"arm {arm!r} not in store; have {sorted(g['solved'])}")
        shape = g["solved"][arm].shape
        return Outcome(
            name=f"always_{arm}",
            solved=g["solved"][arm],
            cost=g["cost"][arm],
            latency=g["latency"][arm],
            action=np.full(shape, arm, dtype=object),
        )

    return policy


def random_route(
    p_large: float, *, small: str = "small", large: str = "large", seed: int = 0
) -> Callable[[Rollouts], Outcome]:
    """Escalate with fixed probability, ignoring the task.

    Controls for "any routing at all helps". A learned router that fails to beat
    this has learned nothing about tasks — it has only learned a mixing ratio,
    which is a single number anyone can tune.
    """

    def policy(store: Rollouts) -> Outcome:
        g = _grids(store)
        rng = np.random.default_rng(seed)
        pick_large = rng.uniform(size=g["solved"][small].shape) < p_large
        return _select(g, pick_large, small, large, f"random_route(p={p_large:g})")

    return policy


def oracle_router(
    *, small: str = "small", large: str = "large"
) -> Callable[[Rollouts], Outcome]:
    """Pick the cheapest arm that actually solves the task.

    Reads the hidden outcome, so it is **not a competitor** — it is the headroom
    a perfect router could capture. Reporting a policy without it hides whether
    a small gap means the policy is weak or the problem is nearly saturated.
    """

    def policy(store: Rollouts) -> Outcome:
        g = _grids(store)
        # Escalate only when small fails and large succeeds. Escalating on a
        # task neither arm solves would spend money for nothing, and an oracle
        # does not do that — including those tasks in the ceiling would make
        # the ceiling unreachable for reasons unrelated to routing.
        pick_large = (g["solved"][small] < 1.0) & (g["solved"][large] >= 1.0)
        return _select(g, pick_large, small, large, "oracle_router")

    return policy


def _select(
    g: dict[str, dict[str, np.ndarray]],
    pick_large: np.ndarray,
    small: str,
    large: str,
    name: str,
) -> Outcome:
    """Non-sequential selection: one arm chosen, only that arm paid for."""
    return Outcome(
        name=name,
        solved=np.where(pick_large, g["solved"][large], g["solved"][small]),
        cost=np.where(pick_large, g["cost"][large], g["cost"][small]),
        latency=np.where(pick_large, g["latency"][large], g["latency"][small]),
        action=np.where(pick_large, large, small).astype(object),
    )


def verifier_gated_cascade(
    *, small: str = "small", large: str = "large", threshold: float = 1.0
) -> Callable[[Rollouts], Outcome]:
    """Run small, execute the visible tests, escalate on failure.

    **The baseline to beat.** It is strong for a structural reason: observing
    failure beats predicting it. A router deciding before generation has a
    prompt; the cascade has an executed test result.

    Its weakness is latency, and only in the tail. On escalated tasks it pays
    small *plus* large serially, so its p95 is roughly the sum. That is where a
    router can genuinely win by skipping a doomed first attempt — and it is
    invisible in a mean.
    """

    def policy(store: Rollouts) -> Outcome:
        g = _grids(store)
        visible = g["visible"][small]
        # NaN visible fraction means ungraded; treat as failure and escalate
        # rather than silently accepting an unverified candidate.
        escalate = ~(visible >= threshold)

        return Outcome(
            name="verifier_gated_cascade",
            solved=np.where(escalate, g["solved"][large], g["solved"][small]),
            # Serial: escalation pays for both arms.
            cost=g["cost"][small] + np.where(escalate, g["cost"][large], 0.0),
            latency=g["latency"][small] + np.where(escalate, g["latency"][large], 0.0),
            action=np.where(escalate, f"{small}->{large}", small).astype(object),
        )

    return policy


def best_of_n_small(
    n: int = 3, *, small: str = "small", threshold: float = 1.0
) -> Callable[[Rollouts], Outcome]:
    """Spend n cheap samples, keep the first that passes the visible tests.

    Controls for "more samples helps". Without it, a router that escalates
    rarely can look like it beat the large arm when it merely benefited from
    extra sampling — a confound that has nothing to do with routing.

    Consumes every seed, so it yields one replicate per task, not n.
    """

    def policy(store: Rollouts) -> Outcome:
        g = _grids(store)
        visible = g["visible"][small][:, :n]
        solved = g["solved"][small][:, :n]
        cost = g["cost"][small][:, :n]
        latency = g["latency"][small][:, :n]

        passes = visible >= threshold
        any_pass = passes.any(axis=1)
        first = np.where(any_pass, passes.argmax(axis=1), n - 1)
        rows = np.arange(solved.shape[0])

        # Charged for every sample drawn up to and including the accepted one —
        # samples are drawn sequentially and you cannot un-spend the failures.
        drawn = first + 1
        total_cost = np.array([cost[i, :d].sum() for i, d in enumerate(drawn)])
        total_latency = np.array([latency[i, :d].sum() for i, d in enumerate(drawn)])

        return Outcome(
            name=f"best_of_{n}_small",
            solved=solved[rows, first][:, None],
            cost=total_cost[:, None],
            latency=total_latency[:, None],
            action=np.full((solved.shape[0], 1), f"best_of_{n}", dtype=object),
            consumes_all_seeds=True,
        )

    return policy


def from_decisions(
    actions: dict[str, str], name: str, *, small: str = "small", large: str = "large"
) -> Callable[[Rollouts], Outcome]:
    """Replay an externally-chosen action per task.

    This is how R3's `decisions.parquet` and R4's tuned heuristic both enter the
    comparison — as a mapping from task to arm, replayed against the same logged
    store as every baseline. Identical accounting for all of them is what makes
    the comparison fair.
    """

    def policy(store: Rollouts) -> Outcome:
        g = _grids(store)
        tasks = store.ordered_tasks
        missing = [t for t in tasks if t not in actions]
        if missing:
            raise KeyError(
                f"{name}: no action for {len(missing)} tasks (e.g. {missing[0]}). "
                f"A policy must decide for every task; defaulting silently would "
                f"attribute the default's outcome to the policy."
            )
        chosen = np.array([actions[t] for t in tasks], dtype=object)
        pick_large = (chosen == large)[:, None]
        pick_large = np.broadcast_to(pick_large, g["solved"][small].shape)
        return _select(g, pick_large, small, large, name)

    return policy


def standard_baselines(
    *, small: str = "small", large: str = "large", n_best_of: int = 3
) -> dict[str, Callable[[Rollouts], Outcome]]:
    """The fixed part of the comparison set.

    `heuristic_route` and the learned policies are added by the caller, because
    both require fitting on the validation split first. Everything here needs
    no fitting at all — which is exactly why they are the floor, the ceiling,
    and the controls.
    """
    return {
        "always_small": always(small),
        "always_large": always(large),
        "random_route(0.5)": random_route(0.5, small=small, large=large),
        f"best_of_{n_best_of}_small": best_of_n_small(n_best_of, small=small),
        "verifier_gated_cascade": verifier_gated_cascade(small=small, large=large),
        "oracle_router": oracle_router(small=small, large=large),
    }
