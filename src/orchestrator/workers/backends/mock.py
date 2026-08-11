"""A deterministic backend that needs no GPU, no network, and no API key.

This is not a stub. It is load-bearing in two ways:

**CI.** The sweep runner, the store, resume, sealing, and the cost model are
all exercised end to end on every commit, on a laptop, in under a second. A
pipeline whose only integration test requires a GPU is a pipeline that is
tested when someone remembers.

**Unblocking R3 and R4.** They develop against schema-valid rows from day one.
This backend plants a signal of *known strength*: a latent per-task difficulty
that both rungs are scored against, so the large model solves a correlated
superset of what the small one solves, which is the structure real routing data
has. A policy that cannot recover a planted signal has a bug, and that is worth
finding in week 1 rather than in week 4 against noisy real data.

Everything is derived by hashing `(task_id, arm, seed, params_hash)`. No RNG
state, so results do not depend on evaluation order and a single cell can be
reproduced without replaying the sweep that produced it.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

from .base import Backend, GenRequest, RawGeneration

# Where a task may carry a solution the mock should emit when it "solves".
# Supplying one makes the mock produce genuinely gradeable rows, which is what
# lets R2's grader be tested against this backend too.
MOCK_SOLUTION_KEY = "mock_solution"


def _unit(*parts: str) -> float:
    """A stable float in [0, 1) from the given strings.

    Hash-derived rather than drawn from an RNG so that a cell's outcome depends
    only on its identity — not on how many cells were generated before it, and
    not on batch composition.
    """
    h = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


class MockBackend(Backend):
    """Simulates two model rungs with a planted, correlated skill gap.

    `small_skill` and `large_skill` are the fraction of tasks each rung solves.
    The defaults sit ~14pp apart, comfortably clear of the Phase 0 gate's
    `A_large - A_small >= 8pp`, so a pipeline test asserting the gate exercises
    a passing path rather than a failing one.
    """

    name: str = "mock"
    mode: str = "sweep"

    small_skill: float = 0.46
    large_skill: float = 0.60
    # Chance a generation is cut off at max_tokens. Non-zero on purpose: silent
    # truncation is a real failure mode and the alarm that watches for it needs
    # something to see in tests.
    truncation_rate: float = 0.03
    refusal_rate: float = 0.005
    error_rate: float = 0.005
    # Seed-level jitter around the latent difficulty, so three seeds of one
    # task disagree sometimes. Without it every seed is identical and R4's
    # cluster bootstrap has nothing to cluster over.
    seed_jitter: float = 0.10

    def __init__(self, small_model: str = "mock-small",
                 large_model: str = "mock-large", **kwargs: float) -> None:
        super().__init__(small_model=small_model, large_model=large_model)
        for key, value in kwargs.items():
            if not hasattr(MockBackend, key):
                raise TypeError(f"unknown MockBackend option {key!r}")
            setattr(self, key, value)
        self.solutions: dict[str, str] = {}

    def register_solution(self, task_id: str, code: str) -> None:
        """Teach the mock what a correct answer to a task looks like.

        The sweep runner registers these from task metadata, so a corpus that
        ships reference solutions produces rows a real grader actually passes.
        """
        self.solutions[task_id] = code

    def _skill(self, role: str) -> float:
        return self.small_skill if role == "small" else self.large_skill

    def _generate(self, requests: Sequence[GenRequest],
                  batch_size: int) -> Iterable[RawGeneration]:
        for req in requests:
            yield self._one(req, batch_size)

    def _one(self, req: GenRequest, batch_size: int) -> RawGeneration:
        model = self.model_id(req.model_role)
        ident = (req.task_id, req.arm, str(req.seed), req.params_hash)

        if _unit("error", *ident) < self.error_rate:
            return RawGeneration.failure(
                model, "simulated backend error", mode=self.mode,
                batch_size=batch_size, wall_ms=120.0,
            )
        if _unit("refusal", *ident) < self.refusal_rate:
            return RawGeneration(
                text="", model_id=model, prefill_tokens=self._prefill(req),
                decode_tokens=0, wall_ms=90.0, finish_reason="refusal",
                mode=self.mode, batch_size=batch_size,
            )

        # One latent difficulty per task, shared by both rungs. This is what
        # makes the two arms correlated rather than independent — the large
        # model mostly solves a superset, with genuine discordance at the
        # margin, which is exactly where a router can win or lose.
        difficulty = _unit("difficulty", req.task_id)
        jitter = (_unit("jitter", *ident) - 0.5) * 2.0 * self.seed_jitter
        solved = self._skill(req.model_role) > (difficulty + jitter)

        truncated = _unit("trunc", *ident) < self.truncation_rate
        code = self._code(req, solved=solved and not truncated)
        text = f"```python\n{code}\n```"
        if truncated:
            # A real truncation loses the closing fence, which is what makes
            # the extractor's `fenced_truncated` path reachable in tests.
            text = f"```python\n{code[: max(10, len(code) // 2)]}"

        decode = max(8, int(len(text) / 3.6))
        prefill = self._prefill(req)
        return RawGeneration(
            text=text,
            model_id=model,
            prefill_tokens=prefill,
            decode_tokens=decode,
            # A plausible wall-clock so the cost model has something to fit:
            # linear in tokens, plus deterministic jitter standing in for
            # server-side variance. Without the jitter a fit against these
            # rows would be exactly perfect, and a test that only ever sees
            # R^2 = 1.0 cannot tell a working regression from a broken one.
            # Under `mode="sweep"` this is explicitly not a latency claim.
            wall_ms=(
                40.0 + prefill * 0.08 + decode * 6.0
                + (_unit("wall", *ident) - 0.5) * 60.0
            ),
            finish_reason="length" if truncated else "stop",
            mode=self.mode,
            batch_size=batch_size,
            tokens_exact=True,
        )

    def _prefill(self, req: GenRequest) -> int:
        return max(1, int((len(req.system) + len(req.user)) / 3.6))

    def _code(self, req: GenRequest, *, solved: bool) -> str:
        """Emit a correct solution, or a plausible wrong one.

        The wrong variant is deliberately *runnable* — it imports, defines the
        entrypoint, and returns the wrong answer. A wrong solution that fails
        to parse would exercise the grader's syntax-error path instead of its
        assertion path, and those are different failures.
        """
        registered = self.solutions.get(req.task_id, "")
        if solved and registered:
            return registered
        if registered:
            return (
                registered
                + "\n\n# mock: deliberately wrong variant\n"
                + "_MOCK_WRONG = True\n"
            )

        # Derived from the content hash, never from `hash()`: string hashing is
        # salted per process, so a `hash()`-derived name would differ between
        # the sweep that wrote a row and the test that reproduces it.
        entry = f"solve_{int(_unit('entry', req.task_id) * 9973)}"
        if solved:
            return (
                f"def {entry}(*args, **kwargs):\n"
                f"    # mock: simulated correct solution\n"
                f"    return True\n"
            )
        return (
            f"def {entry}(*args, **kwargs):\n"
            f"    # mock: simulated incorrect solution\n"
            f"    return None\n"
        )
