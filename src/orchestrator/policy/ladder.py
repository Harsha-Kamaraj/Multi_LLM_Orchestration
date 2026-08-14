"""The repair ladder: sequential attempts, and the leak that lives in them.

A ladder is one task-seed's chain of attempts. Step 0 is the original sample;
step 1 is a repair seeded from it, carrying the failed candidate and its
visible-test output back to the model. `parent_rollout_id` links them, and
`rollout_id` folds both `ladder_step` and the parent in, so a repair is never
mistaken for the sample it repairs.

## The fifth leak

`docs/harsha.md` lists five leaks that actually happen. Four were closable
before a ladder existed. This is the fifth:

    A feature from ladder step k+1 used at step k

It is the same mistake as reading the hidden tests, wearing different clothes.
The decision at step 0 is *whether to repair*, and the outcome of the repair is
the thing being predicted — a feature that reads step 1 while deciding at step 0
has read the answer. It will look like an excellent repair policy.

The guard is structural rather than a rule. `Ladder.upto(k)` returns a ladder
that **does not contain** the later steps, so a feature builder handed one
cannot read them by accident. Asking for a truncated step raises rather than
returning `None`, because a silent `None` becomes a zero and a zero becomes a
feature value.

## Cost is cumulative, and that is the whole economics

A ladder's cost at step k is the sum of every step up to and including k. You
cannot un-spend a failed attempt. Charging only for the accepted step is the
single easiest way to make repair look free, and it flatters repair against
escalation precisely where the comparison matters.

R4's `verifier_gated_cascade` already accounts this way for the escalation
baseline. The repair comparison has to match it or the two are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import LeakageError, PolicyError
from .store import RolloutData

#: Arms that occupy a ladder step above zero. `repair_small` is R1's registered
#: name; the check is on `ladder_step` rather than on the name, and this exists
#: only for reporting.
REPAIR_ARMS: frozenset[str] = frozenset({"repair_small", "repair_large"})


class LadderError(PolicyError):
    """A ladder cannot be assembled or read as asked."""


@dataclass(frozen=True)
class LadderNode:
    """One attempt in a chain."""

    rollout_id: str
    task_id: str
    arm: str
    seed: int
    ladder_step: int
    parent_rollout_id: str | None
    row: Mapping[str, Any]


@dataclass(frozen=True)
class Ladder:
    """One task-seed's chain of attempts, ordered by step.

    `visible_upto` records truncation. A ladder that was never truncated has
    `None`, and one truncated at step k refuses step k+1 with a message that
    says *why* it is missing — "this ladder has no step 2" and "you are not
    allowed to see step 2 yet" are different bugs.
    """

    task_id: str
    seed: int
    nodes: tuple[LadderNode, ...]
    visible_upto: int | None = None

    @property
    def root(self) -> LadderNode:
        return self.nodes[0]

    @property
    def root_arm(self) -> str:
        """Which arm started this chain.

        Part of a ladder's identity, not decoration: `(task_id, seed)` is
        shared by every arm's step-0 sample, so keying on it alone merges the
        cheap and expensive attempts into one impossible chain with two roots.
        """
        return self.root.arm

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(node.ladder_step for node in self.nodes)

    @property
    def depth(self) -> int:
        return max(self.steps) if self.nodes else -1

    def at(self, step: int) -> LadderNode:
        for node in self.nodes:
            if node.ladder_step == step:
                return node
        if self.visible_upto is not None and step > self.visible_upto:
            raise LeakageError(
                f"step {step} of task {self.task_id!r} is not observable at "
                f"step {self.visible_upto}. Deciding whether to repair while "
                f"reading the repair's outcome is the same mistake as reading "
                f"the hidden tests — it will look like an excellent policy."
            )
        raise LadderError(
            f"task {self.task_id!r} seed {self.seed} has no step {step}; "
            f"it has {list(self.steps)}"
        )

    def upto(self, step: int) -> "Ladder":
        """The ladder as it was observable at `step`.

        Truncation, not filtering-on-read: the later nodes are absent from the
        returned object, so nothing downstream can reach them however hard it
        tries. Same guarantee the loader gives by removing the label columns.
        """
        if step < 0:
            raise LadderError(f"ladder steps start at 0; got {step}")
        return Ladder(
            task_id=self.task_id,
            seed=self.seed,
            nodes=tuple(n for n in self.nodes if n.ladder_step <= step),
            visible_upto=step,
        )

    def rows_upto(self, step: int) -> tuple[Mapping[str, Any], ...]:
        return tuple(node.row for node in self.upto(step).nodes)

    def cumulative_cost(self, step: int, column: str = "gpu_seconds") -> float:
        """Everything spent to reach `step`, inclusive.

        Missing values raise rather than being skipped. A ladder that silently
        dropped an uncosted step would report repair as cheaper than it is, and
        the whole Phase 2 gate is a comparison of costs.
        """
        total = 0.0
        for node in self.upto(step).nodes:
            value = node.row.get(column)
            if value is None:
                raise LadderError(
                    f"step {node.ladder_step} of task {self.task_id!r} has no "
                    f"{column}, so the ladder's cost is unknown. Pin a costing "
                    f"rather than summing what is present — a partial total "
                    f"understates repair against escalation."
                )
            total += float(value)
        return total


def build_ladders(rows: Iterable[Mapping[str, Any]]) -> dict[str, Ladder]:
    """Group rows into chains, keyed by the `rollout_id` of each chain's root.

    Keyed by root rather than by `(task_id, seed)`, which every arm's step-0
    sample shares — grouping on that merges the cheap and expensive attempts
    into a single impossible ladder with two roots, and the error it produces
    ("two rows at the same step") points at the data rather than at the key.

    A repair whose parent is absent is refused rather than promoted to a root.
    An orphan means the store is incomplete, and treating it as a step-0 sample
    would silently credit a repair's outcome to a direct generation — while
    charging none of the parent's cost.
    """
    rows = list(rows)
    nodes: dict[str, LadderNode] = {}

    for row in rows:
        step = int(row.get("ladder_step") or 0)
        parent = row.get("parent_rollout_id")
        if step > 0 and not parent:
            raise LadderError(
                f"rollout {row['rollout_id']!r} is at ladder step {step} with "
                f"no parent, so nothing says what it was repairing"
            )
        if step == 0 and parent:
            raise LadderError(
                f"rollout {row['rollout_id']!r} is at step 0 and names a "
                f"parent {parent!r}. A root has nothing above it."
            )
        nodes[str(row["rollout_id"])] = LadderNode(
            rollout_id=str(row["rollout_id"]),
            task_id=str(row["task_id"]),
            arm=str(row["arm"]),
            seed=int(row["seed"]),
            ladder_step=step,
            parent_rollout_id=str(parent) if parent else None,
            row=row,
        )

    def root_of(node: LadderNode) -> str:
        seen: set[str] = set()
        while node.parent_rollout_id:
            if node.rollout_id in seen:
                raise LadderError(
                    f"ladder through {node.rollout_id!r} contains a cycle"
                )
            seen.add(node.rollout_id)
            parent = nodes.get(node.parent_rollout_id)
            if parent is None:
                raise LadderError(
                    f"rollout {node.rollout_id!r} names parent "
                    f"{node.parent_rollout_id!r}, which is not in this store. "
                    f"Treating it as a root would credit a repair's outcome to "
                    f"a direct generation and charge none of the parent's cost."
                )
            node = parent
        return node.rollout_id

    by_root: dict[str, list[LadderNode]] = {}
    for node in nodes.values():
        by_root.setdefault(root_of(node), []).append(node)

    ladders: dict[str, Ladder] = {}
    for root_id, chain in by_root.items():
        ordered = tuple(sorted(chain, key=lambda n: n.ladder_step))
        seen = [n.ladder_step for n in ordered]
        if len(set(seen)) != len(seen):
            raise LadderError(
                f"chain rooted at {root_id!r} has two rows at the same ladder "
                f"step: {seen}. One step is one attempt."
            )
        ladders[root_id] = Ladder(
            task_id=ordered[0].task_id, seed=ordered[0].seed, nodes=ordered,
        )
    return ladders


# ---------------------------------------------------------------------------
# Does repair pay for itself?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Strategy:
    """One counterfactual replay: what it solved, and what it spent."""

    name: str
    accuracy: float
    cost: float
    n: int

    def utility(self, lam: float) -> float:
        return self.accuracy - lam * self.cost


@dataclass(frozen=True)
class RepairGate:
    """ROADMAP Phase 2's gate: repair beats escalation in *some* λ region.

    "Some region" is the operative phrase. Repair is cheaper and weaker,
    escalation is dearer and stronger, so which wins is a question about price
    rather than about quality — and a gate stated as "repair is better" would
    have no answer.
    """

    always_small: Strategy
    escalate: Strategy
    repair: Strategy
    lambdas: tuple[float, ...]
    repair_wins: tuple[float, ...]

    @property
    def verdict(self) -> str:
        return "PASS" if self.repair_wins else "FAIL"

    @property
    def delta_accuracy(self) -> float:
        """Repair's accuracy gain over doing nothing."""
        return self.repair.accuracy - self.always_small.accuracy

    @property
    def delta_cost(self) -> float:
        return self.repair.cost - self.always_small.cost

    @property
    def repair_efficiency(self) -> float:
        """Δaccuracy per unit Δcost. Directly comparable to escalation's."""
        return self.delta_accuracy / self.delta_cost if self.delta_cost else float("nan")

    @property
    def escalation_efficiency(self) -> float:
        delta_cost = self.escalate.cost - self.always_small.cost
        if not delta_cost:
            return float("nan")
        return (self.escalate.accuracy - self.always_small.accuracy) / delta_cost

    def summary(self) -> str:
        lines = [
            f"[{self.verdict}] repair vs escalation over {self.repair.n} ladders",
            f"    always_small  acc={self.always_small.accuracy:.4f}  "
            f"cost={self.always_small.cost:.4f}",
            f"    repair        acc={self.repair.accuracy:.4f}  "
            f"cost={self.repair.cost:.4f}",
            f"    escalate      acc={self.escalate.accuracy:.4f}  "
            f"cost={self.escalate.cost:.4f}",
            f"    efficiency (dacc/dcost): repair "
            f"{self.repair_efficiency:.4f}  escalation "
            f"{self.escalation_efficiency:.4f}",
        ]
        if self.repair_wins:
            lines.append(
                f"    repair wins on utility for {len(self.repair_wins)} of "
                f"{len(self.lambdas)} lambdas, "
                f"[{min(self.repair_wins):.6g}, {max(self.repair_wins):.6g}]"
            )
        else:
            lines.append(
                "    escalation dominates at every lambda on the grid: repair "
                "does not pay for itself here. That is a result, not a bug — "
                "one round of repair on a failed cheap sample may simply be "
                "worth less than the large model."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "delta_accuracy": self.delta_accuracy,
            "delta_cost": self.delta_cost,
            "repair_efficiency": self.repair_efficiency,
            "escalation_efficiency": self.escalation_efficiency,
            "repair_wins": list(self.repair_wins),
            "strategies": {
                s.name: {"accuracy": s.accuracy, "cost": s.cost, "n": s.n}
                for s in (self.always_small, self.escalate, self.repair)
            },
        }


def measure_repair_gate(data: RolloutData, *,
                        lambdas: Sequence[float] | None = None,
                        cheap: str = "small",
                        expensive: str = "large",
                        column: str = "gpu_seconds") -> RepairGate:
    """Replay three strategies over the logged ladders and price them.

    Every strategy is charged the cheap sample first, because all three take
    it: the decision to repair or escalate is only reachable *after* the cheap
    attempt has failed. Charging repair for the repair alone would compare it
    against a baseline that never ran.
    """
    from .decide import LAMBDA_GRID

    lambdas = tuple(float(v) for v in (lambdas if lambdas is not None
                                       else LAMBDA_GRID))
    ladders = build_ladders(data.rows)

    by_task_seed_arm: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in data.rows:
        key = (str(row["task_id"]), int(row["seed"]), str(row["arm"]))
        by_task_seed_arm[key] = row

    solved_small: list[float] = []
    solved_repair: list[float] = []
    solved_escalate: list[float] = []
    cost_small: list[float] = []
    cost_repair: list[float] = []
    cost_escalate: list[float] = []

    def solved(row: Mapping[str, Any]) -> bool:
        return data.label_for(str(row["rollout_id"])).solved

    for _, ladder in sorted(ladders.items()):
        # One replay per *cheap* chain: the decision to repair or escalate is
        # only reachable after the cheap attempt, so a chain rooted at the
        # expensive arm is not a decision point and must not be counted as one.
        if ladder.root_arm != cheap:
            continue
        task_id, seed = ladder.task_id, ladder.seed
        base = ladder.root.row
        big = by_task_seed_arm.get((task_id, seed, expensive))
        if big is None:
            continue

        base_solved = solved(base)
        base_cost = float(base[column])

        solved_small.append(float(base_solved))
        cost_small.append(base_cost)

        if base_solved:
            # Nothing escalates or repairs a success, so all three strategies
            # agree here and are charged identically.
            solved_repair.append(1.0)
            solved_escalate.append(1.0)
            cost_repair.append(base_cost)
            cost_escalate.append(base_cost)
            continue

        # Escalation: the cheap attempt is already spent.
        solved_escalate.append(float(solved(big)))
        cost_escalate.append(base_cost + float(big[column]))

        repair_node = next(
            (n for n in ladder.nodes if n.ladder_step == 1), None
        )
        if repair_node is None:
            # No repair was logged for this failure: the strategy has nothing
            # to do, so it keeps the failure and pays only what it spent.
            solved_repair.append(0.0)
            cost_repair.append(base_cost)
        else:
            solved_repair.append(float(solved(repair_node.row)))
            cost_repair.append(ladder.cumulative_cost(1, column))

    if not solved_small:
        raise LadderError(
            "no task-seed had both arms logged, so no strategy can be replayed"
        )

    def strategy(name: str, solved_list, cost_list) -> Strategy:
        return Strategy(name=name, accuracy=float(np.mean(solved_list)),
                        cost=float(np.mean(cost_list)), n=len(solved_list))

    always_small = strategy("always_small", solved_small, cost_small)
    repair = strategy("repair_on_failure", solved_repair, cost_repair)
    escalate = strategy("escalate_on_failure", solved_escalate, cost_escalate)

    wins = tuple(
        lam for lam in lambdas
        if repair.utility(lam) > escalate.utility(lam)
    )
    return RepairGate(
        always_small=always_small,
        escalate=escalate,
        repair=repair,
        lambdas=lambdas,
        repair_wins=wins,
    )
