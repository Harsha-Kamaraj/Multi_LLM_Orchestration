"""Synthetic rollouts with a planted signal and a known-optimal policy.

This is not a stopgap for missing data. It is the correctness test for every
piece of code that consumes a rollout store, and it is the reason R3 and R4 are
not blocked on a GPU.

The construction is deliberate. Each task gets a latent difficulty `d`. Arm
success is a monotone function of `d`, so an oracle over `d` is the best any
router could possibly do, and we can compute it exactly. Two observable feature
families are generated with *tunable* noise:

* `x_d0` — a prompt-only proxy for `d`, noisy on purpose. This is the D0 signal.
* `visible_passed` — the outcome of running the visible tests on the generated
  candidate. Strongly informative, because observing failure beats predicting
  it. This is the D1 signal.

Because the generator knows `d`, it can emit the **exact optimal action** at any
λ. That turns a vague "does the policy look reasonable?" into a sharp assertion:
the evaluator must rank the planted-optimal policy first, and its confidence
interval must cover the planted effect size. A statistics harness that has never
been run against a known answer is not validated — it is merely untested code
that produces numbers.

Nothing here imports anything R1 owns. The generator writes rows that satisfy
`rollout.schema.json`, and that shared contract is the only coupling.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from .version import SCHEMA_VERSION

ARMS: tuple[str, ...] = ("small", "large")


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class ArmSpec:
    """One arm's planted behaviour.

    `skill` shifts the success curve up; `cost` and `latency` are the price of
    pulling it. Success is `sigmoid(skill - slope * d)`, so a higher-skill arm
    dominates everywhere and the interesting question is never *whether* to
    escalate but *when it is worth paying for*.
    """

    name: str
    skill: float
    cost: float
    latency: float
    slope: float = 6.0
    # Spread of the per-row cost noise, as a fraction of `cost`. Costs in a real
    # store vary with output length; a generator emitting a constant would let a
    # buggy matched-cost comparison pass.
    cost_cv: float = 0.25

    def p_solve(self, d: np.ndarray) -> np.ndarray:
        return _sigmoid(self.skill - self.slope * d)


DEFAULT_ARMS: tuple[ArmSpec, ...] = (
    # ~20pp apart on average, matching the 1.5B/7B gap the project actually
    # runs, and comfortably past the 8pp Phase 0 gate.
    ArmSpec("small", skill=1.4, cost=1.0, latency=1.0),
    ArmSpec("large", skill=3.2, cost=6.0, latency=3.4),
)


@dataclass(frozen=True)
class SynthConfig:
    """Every knob that changes the data, in one hashable place.

    The config hash goes into `run_id`, so two fixture sets built from different
    settings can never collide in a store — the same discipline the real sweep
    uses, for the same reason.
    """

    n_tasks: int = 400
    seeds: int = 3
    arms: tuple[ArmSpec, ...] = DEFAULT_ARMS
    # Correlation between the prompt-only proxy and true difficulty. 0.40 lands
    # AUC_D0 in 0.62–0.69, which is the band ROADMAP.md predicts for D0 — weak,
    # and deliberately so. A fixture that flattered pre-generation routing would
    # hide the D0/D1 asymmetry that is the project's central finding.
    d0_signal: float = 0.40
    # How faithfully visible tests reflect the hidden outcome. High but not
    # perfect: if it were 1.0 the cascade would be unbeatable by construction.
    d1_fidelity: float = 0.82
    visible_tests: int = 4
    hidden_tests: int = 8
    date: str = "2026-01-01"
    git_sha7: str = "0f1ced0"

    @property
    def config_hash6(self) -> str:
        payload = "|".join(
            [
                str(self.n_tasks), str(self.seeds), str(self.d0_signal),
                str(self.d1_fidelity), str(self.visible_tests),
                str(self.hidden_tests),
                *(f"{a.name}:{a.skill}:{a.cost}:{a.latency}:{a.slope}"
                  for a in self.arms),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:6]

    @property
    def run_id(self) -> str:
        return f"{self.date}-{self.git_sha7}-{self.config_hash6}"


@dataclass
class SynthResult:
    """Rows plus the ground truth that makes them testable."""

    rows: list[dict[str, Any]]
    truth: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.rows[0]["run_id"] if self.rows else ""

    def optimal_actions(self, lam: float) -> dict[str, str]:
        """The exact best action per task at cost weight `lam`.

        Computed from latent difficulty, which no policy can observe. This is
        the ceiling, and the yardstick a learned policy is measured against.
        """
        d = self.truth["difficulty"]
        task_ids = self.truth["task_ids"]
        specs: Sequence[ArmSpec] = self.truth["arms"]
        utilities = np.stack([a.p_solve(d) - lam * a.cost for a in specs])
        best = utilities.argmax(axis=0)
        return {tid: specs[i].name for tid, i in zip(task_ids, best)}

    def optimal_value(self, lam: float) -> float:
        """Expected utility of the optimal policy at `lam` — the planted ceiling."""
        d = self.truth["difficulty"]
        specs: Sequence[ArmSpec] = self.truth["arms"]
        utilities = np.stack([a.p_solve(d) - lam * a.cost for a in specs])
        return float(utilities.max(axis=0).mean())

    def planted_auc_d0(self) -> float:
        """The AUC an ideal D0 model would reach on this fixture.

        Not a target to tune toward — a bound. A feature builder reporting more
        than this has leaked something.
        """
        return _auc(self.truth["x_d0"], self.truth["small_solved_any"])


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, ties averaged. No sklearn dependency."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks within tie groups so a constant score scores 0.5, not 1.0.
    sorted_scores = np.asarray(scores)[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def generate(config: SynthConfig | None = None, *, seed: int = 0) -> SynthResult:
    """Build a fixture store with a planted, measurable signal."""
    cfg = config or SynthConfig()
    rng = np.random.default_rng(seed)
    n = cfg.n_tasks

    # Latent difficulty. Nothing observable exposes this directly.
    difficulty = rng.uniform(0.0, 1.0, size=n)

    # The prompt-only proxy. Oriented so that *higher means more likely to
    # solve* — the same direction as the label. An inverted proxy is a real
    # bug that shows up as AUC below 0.5, and a feature builder that "fixes" it
    # by flipping the sign has fit the label rather than the feature.
    # Standardized so the correlation is the knob rather than an accident of
    # scale, which keeps the planted AUC stable when n changes.
    noise = rng.normal(size=n)
    rho = cfg.d0_signal
    x_d0 = rho * _standardize(-difficulty) + math.sqrt(max(0.0, 1 - rho**2)) * noise

    task_ids = [f"synth/{i:05d}" for i in range(n)]
    splits = _assign_splits(rng, n)

    rows: list[dict[str, Any]] = []
    solved: dict[str, np.ndarray] = {}

    for spec in cfg.arms:
        p = spec.p_solve(difficulty)
        # (n_tasks, seeds) — independent draws, so seed variance is real and a
        # bootstrap that ignores the seed cluster will visibly understate width.
        draws = rng.uniform(size=(n, cfg.seeds)) < p[:, None]
        solved[spec.name] = draws

        for t in range(n):
            for s in range(cfg.seeds):
                rows.append(
                    _row(cfg, rng, spec, task_ids[t], splits[t], s,
                         bool(draws[t, s]), float(x_d0[t]),
                         float(difficulty[t]))
                )

    return SynthResult(
        rows=rows,
        truth={
            "config": cfg,
            "arms": cfg.arms,
            "task_ids": task_ids,
            "difficulty": difficulty,
            "x_d0": x_d0,
            "splits": splits,
            "solved": solved,
            "small_solved_any": solved["small"].any(axis=1),
        },
    )


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _assign_splits(rng: np.random.Generator, n: int) -> list[str]:
    """60/20/20 at task level. Never at row level — every seed of one task must
    land in the same split, or the test set contains near-copies of training
    rows and every interval is too narrow."""
    idx = rng.permutation(n)
    out = [""] * n
    a, b = int(0.6 * n), int(0.8 * n)
    for rank, i in enumerate(idx):
        out[i] = "train" if rank < a else ("val" if rank < b else "test")
    return out


def _row(
    cfg: SynthConfig,
    rng: np.random.Generator,
    spec: ArmSpec,
    task_id: str,
    split: str,
    seed: int,
    hidden_solved: bool,
    x_d0: float,
    difficulty: float,
) -> dict[str, Any]:
    """One schema-valid rollout row."""
    # Visible tests agree with the hidden outcome `d1_fidelity` of the time. The
    # disagreement is what leaves room for a router to beat the cascade.
    faithful = rng.uniform() < cfg.d1_fidelity
    visible_all_pass = hidden_solved if faithful else not hidden_solved
    visible_passed = (
        cfg.visible_tests if visible_all_pass
        else int(rng.integers(0, cfg.visible_tests))
    )
    hidden_passed = (
        cfg.hidden_tests if hidden_solved
        else int(rng.integers(0, cfg.hidden_tests))
    )

    decode = int(max(1, rng.normal(220, 60)))
    prefill = int(max(1, rng.normal(180, 40)))
    gpu_seconds = float(spec.cost * rng.lognormal(0.0, spec.cost_cv))
    latency = float(spec.latency * rng.lognormal(0.0, spec.cost_cv))

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg.run_id,
        "task_id": task_id,
        "arm": spec.name,
        "seed": seed,
        "params_hash": hashlib.sha256(
            f"{spec.name}|{cfg.config_hash6}".encode()
        ).hexdigest()[:12],
        "split": split,
        "dataset": "synth",
        "code_version": cfg.git_sha7,

        "text": "",
        "code": f"def solve():  # {task_id} {spec.name} s{seed}\n    ...\n",
        "model_id": f"synthetic/{spec.name}",
        "prefill_tokens": prefill,
        "decode_tokens": decode,
        # Deliberately unlike `imputed_latency_s`: a consumer that reads this as
        # latency should produce visibly wrong numbers in the fixture, not
        # plausible ones.
        "wall_ms": float(rng.uniform(50, 4000)),
        "finish_reason": "stop",

        "mode": "sweep",
        "batch_size": 32,
        "backend": "synthetic",
        "tokens_exact": True,

        "extract_strategy": "fence",
        "code_parses": True,

        "ladder_step": 0,
        "parent_rollout_id": None,

        "gpu_seconds": gpu_seconds,
        "imputed_latency_s": latency,

        "visible_passed": visible_passed,
        "visible_total": cfg.visible_tests,
        "hidden_passed": hidden_passed,
        "hidden_total": cfg.hidden_tests,
        "error_class": None,
        "hack_flags": [],
        "grade_duration_s": float(rng.uniform(0.05, 1.5)),

        "created_at": f"{cfg.date}T00:00:00Z",
        "error": None,
        # The latent value rides along under `extra` so tests can assert against
        # it. A feature builder that reads it is leaking, and R4's leakage audit
        # checks for exactly this key.
        "extra": {"_synth_difficulty": difficulty, "_synth_x_d0": x_d0},
    }
    row["rollout_id"] = _rollout_id(row)
    return row


def _rollout_id(row: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for part in (row["run_id"], row["task_id"], row["arm"], str(row["seed"]),
                 row["params_hash"], str(row["ladder_step"]),
                 row.get("parent_rollout_id") or ""):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def iter_rows(config: SynthConfig | None = None, *, seed: int = 0) -> Iterator[dict]:
    """Convenience for tests that only need rows."""
    yield from generate(config, seed=seed).rows
