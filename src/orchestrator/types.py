"""Core data types shared across orchestration, grading, and evaluation.

Everything that crosses a module boundary is one of these. They are all
JSON-serializable so a rollout can be written to disk and replayed later
without re-running any model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Action = Literal["direct_small", "escalate_large", "decompose"]

ACTIONS: tuple[Action, ...] = ("direct_small", "escalate_large", "decompose")


@dataclass(frozen=True)
class Task:
    """One unit of work with an executable correctness check.

    `tests` is a pytest module source string. It is executed against the
    model's solution, so it must import the solution by module name
    (`solution`), not by pasting code inline.
    """

    task_id: str
    prompt: str
    tests: str
    entrypoint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Task":
        return Task(
            task_id=d["task_id"],
            prompt=d["prompt"],
            tests=d["tests"],
            entrypoint=d.get("entrypoint", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass(frozen=True)
class Usage:
    """Token accounting for a single model call."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Grade:
    """Result of running the executable checks. No model is involved."""

    passed: int
    total: int
    error: str | None = None
    stdout: str = ""
    duration_s: float = 0.0

    @property
    def score(self) -> float:
        """Fraction of checks that passed, in [0, 1]."""
        if self.total <= 0:
            return 0.0
        return self.passed / self.total

    @property
    def solved(self) -> bool:
        """Strict all-or-nothing correctness — the headline metric."""
        return self.total > 0 and self.passed == self.total


@dataclass
class ArmResult:
    """What one arm produced for one task, with full cost/latency accounting."""

    task_id: str
    action: Action
    output: str
    grade: Grade
    cost_usd: float
    latency_s: float
    usages: list[Usage] = field(default_factory=list)
    n_model_calls: int = 0
    error: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["action"] = str(self.action)
        return json.dumps(d)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArmResult":
        return ArmResult(
            task_id=d["task_id"],
            action=d["action"],
            output=d["output"],
            grade=Grade(**d["grade"]),
            cost_usd=d["cost_usd"],
            latency_s=d["latency_s"],
            usages=[Usage(**u) for u in d.get("usages", [])],
            n_model_calls=d.get("n_model_calls", 0),
            error=d.get("error"),
        )
