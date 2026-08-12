"""The grader interface, and the `Grade` it returns.

A grader turns `(task, code)` into a `Grade` by *executing* something —
running tests in a sandbox. No grader in this package calls a model. That is
the point: the correctness signal the policy optimizes against must not
itself be a model's opinion.

**Grading is a pure function.** Same `(task, code)` in, same `Grade` out, on
any machine, in any order. `duration_s` is the one field allowed to vary
between runs of the same input — it's a measurement of the run, not part of
its identity.

**Extraction is R1's, not ours.** A grader here receives already-extracted
`code`, never raw model output. Parsing markdown fences would make a prompt
change on R1's side a bug in this file; that dependency was removed on
purpose. See `guru.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..types import Task

# Every `Grade` this package can produce carries one of these. `"none"` means
# the harness ran cleanly — it says nothing about whether the solution was
# *correct*, only that nothing crashed collecting or running it. An ordinary
# wrong answer is `error_class="none"` with `passed < total`; that is the
# normal graded outcome, not an error.
ERROR_CLASSES = (
    "none",           # harness ran cleanly; passed/total reflects the outcome
    "empty_code",     # nothing to grade — no sandbox spent
    "syntax_error",   # code.py fails ast.parse — no sandbox spent
    "timeout",        # a run exceeded the configured timeout
    "harness_error",  # the in-sandbox runner never produced a report
    "runtime_error",  # an exception at collection/import time, not an assertion
)


@dataclass(frozen=True)
class TestResult:
    """The outcome of running one test tier (visible, or hidden) once.

    `errors` is truncated failure text for debugging. It is not part of the
    result's identity — two runs of the same `(task, code)` can differ in the
    exact traceback text (line numbers in a temp path, for instance) without
    the grade itself differing.
    """

    __test__ = False  # not a pytest test class — it just starts with "Test"

    passed: int
    total: int
    errors: str = ""

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def rate(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.passed / self.total


@dataclass(frozen=True)
class Grade:
    """Result of running the executable checks. No model is involved.

    `visible` and `hidden` are separate fields with separate accessors on
    purpose — enforcing the visible/hidden split in code, not by convention.
    There is deliberately no single scalar that blends the two tiers; a
    feature builder that wants "did it pass" has to name which tier it means.

    `solved` is the one property derived from `hidden` alone. It is the
    project's headline correctness label — and a label only. Reading it from
    a D0/D1 feature builder is the leak diya.md and harsha.md both warn about.
    """

    visible: TestResult
    hidden: TestResult
    error_class: str = "none"
    hack_flags: tuple[str, ...] = field(default_factory=tuple)
    duration_s: float = 0.0
    stdout: str = ""

    def __post_init__(self) -> None:
        if self.error_class not in ERROR_CLASSES:
            raise ValueError(
                f"error_class {self.error_class!r} not in {ERROR_CLASSES}"
            )

    # -- flat accessors matching Generation.to_row()'s column names ----------

    @property
    def visible_passed(self) -> int:
        return self.visible.passed

    @property
    def visible_total(self) -> int:
        return self.visible.total

    @property
    def hidden_passed(self) -> int:
        return self.hidden.passed

    @property
    def hidden_total(self) -> int:
        return self.hidden.total

    @property
    def solved(self) -> bool:
        """Hidden all-pass. LABEL ONLY — never read by a feature builder."""
        return self.hidden.all_passed

    def to_row(self) -> dict[str, object]:
        """Project onto the null grading keys `Generation.to_row()` emits.

        Exactly `visible_passed, visible_total, hidden_passed, hidden_total,
        error_class, hack_flags, grade_duration_s` — the columns R1's row
        already carries, present but null until a grader fills them in.
        """
        return {
            "visible_passed": self.visible_passed,
            "visible_total": self.visible_total,
            "hidden_passed": self.hidden_passed,
            "hidden_total": self.hidden_total,
            "error_class": self.error_class,
            "hack_flags": list(self.hack_flags),
            "grade_duration_s": self.duration_s,
        }


class Grader(Protocol):
    def grade(self, task: Task, code: str) -> Grade: ...
