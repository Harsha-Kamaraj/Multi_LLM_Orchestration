"""The decision rule, swept across λ, written as `decisions.parquet`.

    choose(x, lam) = argmax_a  P_pass[a](x) - lam * E_cost[a](x)

λ enters here and nowhere else. The heads were fitted without ever having met
it, so one training run produces the entire frontier and λ is a dial someone
can turn in production without refitting anything. That is the whole reason
`heads.py` refuses to learn a scalarized reward.

## The grid is frozen

`LAMBDA_GRID` is a fixed, log-spaced range in units of *inverse GPU-seconds*,
written down before any of it was run. A grid chosen after seeing the frontier
is a knob, and the specific way it goes wrong is subtle: widening the range
until the curve looks good is indistinguishable, in the report, from a curve
that was always that shape.

Log spacing because the interesting region spans orders of magnitude. λ = 0.001
and λ = 0.002 are the same policy; λ = 0.001 and λ = 1.0 are not.

The endpoints are deliberately outside the useful range, so the sweep contains
the degenerate policies at both ends — everything routed to the cheap arm, and
everything routed to whichever arm predicts best. A frontier whose extremes are
*not* degenerate is a frontier that has been cropped.

## Why this is D0 only

`from_decisions` in R4's `eval/policies.py` replays one action per task and
charges for that arm. That is exactly a D0 routing decision.

A D1 policy decides something different — *whether to escalate*, after the
cheap arm has already generated. Replaying that as "the action was `large`"
would charge for the large arm alone and silently discard the small arm's cost,
which is already spent and cannot be un-spent. The cascade baselines account
for this correctly; `from_decisions` has no way to express it. So D1 is refused
here rather than emitted in a shape that would quietly understate its cost, and
the gap is recorded for R4.

## What this cannot produce

Decisions for the test split. `store.load_rollouts` will not open it, so a
sweep covers only the splits R3 can see. R4 applies `policy.pkl` to test
themselves, once, after pre-registration — which is the arrangement, not a
limitation to work around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import PolicyError
from .heads import PolicyHeads
from .store import RolloutData

#: The frozen sweep: 121 points from 1e-4 to 1e2 inverse GPU-seconds, which is
#: six decades at twenty points per decade.
#:
#: Read λ as "how many units of pass-probability one GPU-second is worth". At
#: 1e-4 a whole GPU-second buys 0.0001 of probability, so cost is irrelevant and
#: the rule is argmax P_pass. At 1e2 a hundredth of a GPU-second outweighs
#: certainty, so the rule is argmax -cost. Everything interesting is between.
#:
#: The density is not arbitrary. On fixtures the region where tasks route both
#: ways spans about a fifth of a decade — every task has a similar predicted
#: quality gap between the arms, so they nearly all flip at once. At four
#: points per decade that whole region collapses to a single λ and the frontier
#: is one cliff; at twenty it resolves. Chosen against fixtures, before any real
#: data was read, and frozen — if it proves too coarse on the pilot, widening it
#: is a reviewed change to this line rather than a quiet retune.
LAMBDA_GRID: tuple[float, ...] = tuple(
    float(v) for v in np.logspace(-4.0, 2.0, 121)
)

DECISIONS_PARQUET = "decisions.parquet"
DECISIONS_JSONL = "decisions.jsonl"


class DecisionError(PolicyError):
    """A decision cannot be made or written as asked."""


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmScore:
    """One arm's three predictions for one task, before λ is applied."""

    arm: str
    p_pass: float
    e_cost: float
    e_latency: float

    def utility(self, lam: float) -> float:
        return self.p_pass - lam * self.e_cost


def choose(scores: Sequence[ArmScore], lam: float) -> ArmScore:
    """Highest utility wins; ties go to the cheaper arm.

    Ties are not hypothetical — at large λ every arm's utility is dominated by
    cost, and two arms with equal predicted cost tie exactly. Breaking toward
    the cheaper arm makes the sweep deterministic and makes the degenerate end
    of the frontier the *cheap* end, which is the one a reader expects to see
    there.
    """
    if not scores:
        raise DecisionError("no arms to choose between")
    best = max(scores, key=lambda s: (s.utility(lam), -s.e_cost, s.arm))
    return best


# ---------------------------------------------------------------------------
# One task, every arm
# ---------------------------------------------------------------------------


def _one_row_per_task(rows: Sequence[Mapping[str, Any]],
                      arm: str) -> dict[str, Mapping[str, Any]]:
    """A representative row per task for one arm.

    Legitimate only at D0, where every feature is task-level and all seeds of a
    task therefore produce identical predictions. `score_tasks` refuses D1 for
    this reason among others.
    """
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row["arm"]) != arm:
            continue
        out.setdefault(str(row["task_id"]), row)
    return out


def score_tasks(policy: PolicyHeads, data: RolloutData,
                coefficients: Any, *,
                split: str | None = None) -> dict[str, tuple[ArmScore, ...]]:
    """Every arm's three predictions, per task. λ is not involved yet."""
    if policy.decision_point != "D0":
        raise DecisionError(
            f"this policy decides at {policy.decision_point}, and a "
            f"{policy.decision_point} decision is 'escalate or not', not "
            f"'which arm'. Replaying it through `from_decisions` would charge "
            f"for the escalated arm alone and drop the cheap arm's cost, which "
            f"was already spent. Use a cascade accounting instead."
        )

    rows = [r for r in data.rows
            if split is None or str(r.get("split")) == split]
    if not rows:
        raise DecisionError(
            f"run {data.run_id} has no rows"
            + (f" in split {split!r}" if split else "")
        )

    per_arm: dict[str, dict[str, Mapping[str, Any]]] = {
        arm: _one_row_per_task(rows, arm) for arm in policy.arm_names
    }
    tasks = sorted(set.intersection(*(set(v) for v in per_arm.values())))
    if not tasks:
        raise DecisionError(
            "no task was swept on every arm, so no task can be routed. A "
            "policy that decides between arms needs both of them observed."
        )

    scored: dict[str, tuple[ArmScore, ...]] = {}
    for arm in policy.arm_names:
        ordered = [per_arm[arm][task] for task in tasks]
        head = policy.arms[arm]
        p = head.p_pass(ordered, policy.features)
        cost = head.e_cost(ordered, policy.features, coefficients)
        latency = head.e_latency(ordered, policy.features, coefficients)
        for i, task in enumerate(tasks):
            scored.setdefault(task, ())
            scored[task] = scored[task] + (ArmScore(
                arm=arm,
                p_pass=float(p[i]),
                e_cost=float(cost[i]),
                e_latency=float(latency[i]),
            ),)
    return scored


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sweep:
    """Every decision at every λ, plus what it took to make them."""

    run_id: str
    decision_point: str
    policy_name: str
    publishable: bool
    lambdas: tuple[float, ...]
    #: Long format: one record per (λ, task, arm), with `chosen` marking the
    #: winner. Long rather than wide so the schema does not change when an arm
    #: is added, and so every number behind a decision is auditable rather than
    #: only the one that won.
    records: tuple[dict[str, Any], ...]

    def actions(self, lam: float) -> dict[str, str]:
        """`{task_id: arm}` for one λ — exactly R4's `from_decisions` input."""
        if lam not in self.lambdas:
            raise DecisionError(
                f"lambda {lam!r} is not on the frozen grid. Interpolating "
                f"between swept points would report a policy that was never "
                f"run; add it to LAMBDA_GRID and re-sweep."
            )
        return {
            record["task_id"]: record["arm"]
            for record in self.records
            if record["lam"] == lam and record["chosen"]
        }

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted({r["task_id"] for r in self.records}))

    def arm_counts(self, lam: float) -> dict[str, int]:
        """How many tasks each arm was chosen for at one λ."""
        counts = {arm: 0 for arm in sorted({r["arm"] for r in self.records})}
        for arm in self.actions(lam).values():
            counts[arm] += 1
        return counts

    def arm_share(self, lam: float) -> dict[str, float]:
        """Fraction of tasks routed to each arm at one λ.

        Counted as integers and divided once. Accumulating `1/n` per task
        instead leaves a fully degenerate split at 0.9999999999999999, which
        reads as "not degenerate" to any exact comparison — so
        `degenerate_lambdas` would have found nothing, ever, and the frontier
        would have looked healthy at both ends.
        """
        counts = self.arm_counts(lam)
        total = sum(counts.values()) or 1
        return {arm: n / total for arm, n in counts.items()}

    def degenerate_lambdas(self) -> tuple[float, ...]:
        """λ values that route every task to one arm.

        Reported rather than dropped. The frontier is supposed to contain its
        own degenerate endpoints — a sweep with none has been cropped, and a
        sweep that is degenerate everywhere means the grid missed the region
        where cost and quality actually trade.
        """
        return tuple(
            lam for lam in self.lambdas
            if max(self.arm_counts(lam).values()) == sum(self.arm_counts(lam).values())
        )

    def mixed_lambdas(self) -> tuple[float, ...]:
        """λ values that actually route some tasks each way.

        These are the only points on the frontier that describe a *policy*
        rather than a constant. A sweep with one or two of them has not
        resolved the trade-off region, whatever its endpoints look like.
        """
        degenerate = set(self.degenerate_lambdas())
        return tuple(lam for lam in self.lambdas if lam not in degenerate)

    def summary(self) -> str:
        lines = [
            f"sweep for {self.decision_point} from run {self.run_id}"
            + ("" if self.publishable else "  [not publishable]"),
            f"  {len(self.tasks)} tasks x {len(self.lambdas)} lambdas",
        ]
        for lam in self.lambdas:
            share = self.arm_share(lam)
            rendered = "  ".join(f"{arm}={value:.0%}"
                                 for arm, value in sorted(share.items()))
            lines.append(f"  lambda={lam:<12.6g} {rendered}")
        mixed = self.mixed_lambdas()
        if not mixed:
            lines.append(
                "  every lambda is degenerate: the grid never crosses the "
                "point where cost and quality trade. Widen it, or the arms "
                "are not actually differentiated."
            )
        elif len(mixed) < 3:
            lines.append(
                f"  only {len(mixed)} lambda(s) route tasks both ways, so the "
                f"trade-off region is under-resolved. Usually this means the "
                f"policy separates tasks weakly: if every task has nearly the "
                f"same quality gap between arms, they all flip at nearly the "
                f"same lambda and the frontier is a cliff rather than a curve."
            )
        return "\n".join(lines)

    def manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decision_point": self.decision_point,
            "policy_name": self.policy_name,
            "publishable": self.publishable,
            "lambdas": list(self.lambdas),
            "n_tasks": len(self.tasks),
            "n_records": len(self.records),
            "degenerate_lambdas": list(self.degenerate_lambdas()),
            "mixed_lambdas": list(self.mixed_lambdas()),
            "arm_share": {
                str(lam): self.arm_share(lam) for lam in self.lambdas
            },
        }


def sweep_lambda(policy: PolicyHeads, data: RolloutData, coefficients: Any, *,
                 lambdas: Sequence[float] = LAMBDA_GRID,
                 split: str | None = None,
                 policy_name: str = "learned_D0") -> Sweep:
    """Decide for every task at every λ. One fitted policy, the whole frontier."""
    scored = score_tasks(policy, data, coefficients, split=split)
    if not lambdas:
        raise DecisionError("an empty lambda grid produces no frontier")

    records: list[dict[str, Any]] = []
    for lam in lambdas:
        for task in sorted(scored):
            scores = scored[task]
            winner = choose(scores, lam)
            for score in scores:
                records.append({
                    "run_id": data.run_id,
                    "policy_name": policy_name,
                    "decision_point": policy.decision_point,
                    "lam": float(lam),
                    "task_id": task,
                    "arm": score.arm,
                    "p_pass": score.p_pass,
                    "e_cost": score.e_cost,
                    "e_latency": score.e_latency,
                    "utility": score.utility(lam),
                    "chosen": score.arm == winner.arm,
                })

    return Sweep(
        run_id=data.run_id,
        decision_point=policy.decision_point,
        policy_name=policy_name,
        publishable=data.publishable,
        lambdas=tuple(float(v) for v in lambdas),
        records=tuple(records),
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_decisions(sweep: Sweep, directory: Path | str) -> Path:
    """Write `decisions.parquet`, with JSONL as the authoritative copy.

    Same split as the rollout store: JSONL is what exists on a host without
    pyarrow, Parquet is what R3 and R4 read. Writing only Parquet would make
    the deliverable unreadable exactly where a sweep is most likely to run.

    The `run_id` is on every record, not only in the manifest. A decisions file
    separated from its manifest still pins the run it came from, and nothing
    anywhere resolves "latest".
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / DECISIONS_JSONL
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in sweep.records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    (directory / "decisions_manifest.json").write_text(
        json.dumps(sweep.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="",
    )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return directory

    pq.write_table(pa.Table.from_pylist(list(sweep.records)),
                   directory / DECISIONS_PARQUET)
    return directory


def read_actions(path: Path | str, lam: float) -> dict[str, str]:
    """Load `{task_id: arm}` for one λ, for handing to R4's `from_decisions`.

    Reads the JSONL, which is the authoritative copy. An exact match on λ is
    required for the same reason `Sweep.actions` requires one.
    """
    actions: dict[str, str] = {}
    seen: set[float] = set()
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        seen.add(float(record["lam"]))
        if float(record["lam"]) == lam and record["chosen"]:
            actions[str(record["task_id"])] = str(record["arm"])
    if not actions:
        raise DecisionError(
            f"no decisions at lambda={lam!r}; the file holds "
            f"{sorted(seen)[:5]}{'...' if len(seen) > 5 else ''}"
        )
    return actions
