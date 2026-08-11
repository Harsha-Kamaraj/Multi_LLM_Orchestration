"""Prompt templates, versioned by content rather than by label.

Prompt format is entirely R1's business — R2 receives extracted code, never raw
model output — so templates live here and nowhere else.

**A template's identity is the hash of its text.** `template_hash` is derived
from the rendered system and user bodies, not from `template_id`. Editing a
template without renaming it still changes the hash, which changes
`params_hash`, which makes old rollouts distinguishable from new ones. That is
the whole mechanism preventing a mid-sweep prompt tweak from silently poisoning
a run, and it only works because the label is not the identity.

Hidden tests must never enter a prompt. `PromptContext` is the only thing a
template can render from, and it carries a visible-tests field and no hidden
one — the guard is structural, not a convention.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..types import Task

# Keys in `Task.metadata` that would leak labels into a prompt if rendered.
# Deliberately narrow: `hidden` as a substring, because in this project no
# innocent metadata key contains it, plus an exact list of reference-solution
# names. A looser rule (matching "solution" anywhere) trips on harmless keys
# like `n_solutions_found` and would block a sweep for no reason.
_FORBIDDEN_METADATA_SUBSTRINGS = ("hidden",)
_FORBIDDEN_METADATA_KEYS = frozenset({
    "canonical_solution", "reference_solution", "solution", "solution_code",
    "answer", "answer_code", "gold", "gold_code",
})

# Where R2 publishes the model-visible portion of the test suite. Absent until
# the visible/hidden split lands; see `PromptContext.from_task`.
VISIBLE_TESTS_KEY = "visible_tests"


@dataclass(frozen=True)
class PromptContext:
    """Everything a template is allowed to see about a task.

    Deliberately narrower than `Task`. A template cannot reach the full test
    suite, because a `Task` in the current scaffold carries visible and hidden
    tests in one undifferentiated `tests` field, and rendering that would put
    labels in the prompt.
    """

    task_id: str
    prompt: str
    entrypoint: str = ""
    visible_tests: str = ""
    # Phase 2 repair ladder: the prior attempt and what the visible tests said
    # about it. Empty for every step-0 generation.
    prior_code: str = ""
    prior_feedback: str = ""

    @staticmethod
    def from_task(task: Task, *, prior_code: str = "",
                  prior_feedback: str = "") -> "PromptContext":
        """Project a `Task` down to what a prompt may see.

        Visible tests are read from `metadata["visible_tests"]` and nowhere
        else. When R2 has not published a split yet the field is absent, and
        the prompt simply carries no tests — it deliberately does *not* fall
        back to `Task.tests`, which is the full suite including the hidden
        cases that serve as labels.
        """
        _assert_no_label_keys(task.metadata)
        return PromptContext(
            task_id=task.task_id,
            prompt=task.prompt,
            entrypoint=task.entrypoint,
            visible_tests=str(task.metadata.get(VISIBLE_TESTS_KEY, "")),
            prior_code=prior_code,
            prior_feedback=prior_feedback,
        )


def _assert_no_label_keys(metadata: dict[str, Any]) -> None:
    """Refuse to build a context from metadata that carries labels.

    Cheap, and it fires at the moment someone adds `hidden_tests` to metadata
    rather than three weeks later when a leakage audit finds it.
    """
    for key in metadata:
        low = str(key).lower()
        if low == VISIBLE_TESTS_KEY:
            continue
        hit = low in _FORBIDDEN_METADATA_KEYS or any(
            bad in low for bad in _FORBIDDEN_METADATA_SUBSTRINGS
        )
        if hit:
            raise ValueError(
                f"task metadata key {key!r} carries a label and must not be "
                f"reachable from a prompt; hidden tests and reference solutions "
                f"belong to R2 and are never inputs to generation"
            )


@dataclass(frozen=True)
class PromptTemplate:
    """A named, content-hashed pair of system and user templates.

    `user` is rendered with `str.format`, so literal braces in the body must
    be doubled. Code fences in the instructions are safe — they contain no
    braces — but anything added later that does must escape them.
    """

    template_id: str
    system: str
    user: str
    # Rendering flags that change the text, and therefore belong in the hash.
    include_visible_tests: bool = True

    def render(self, ctx: PromptContext) -> tuple[str, str]:
        """Return `(system, user)` ready to send to a backend."""
        tests_block = ""
        if self.include_visible_tests and ctx.visible_tests.strip():
            tests_block = (
                "\n\nYour solution must satisfy these tests:\n\n"
                f"```python\n{ctx.visible_tests.strip()}\n```"
            )

        prior_block = ""
        if ctx.prior_code.strip():
            prior_block = (
                "\n\nYour previous attempt:\n\n"
                f"```python\n{ctx.prior_code.strip()}\n```"
            )
        if ctx.prior_feedback.strip():
            prior_block += (
                "\n\nRunning the visible tests against it reported:\n\n"
                f"```\n{ctx.prior_feedback.strip()}\n```"
            )

        user = self.user.format(
            prompt=ctx.prompt.strip(),
            entrypoint=ctx.entrypoint,
            tests_block=tests_block,
            prior_block=prior_block,
        )
        return self.system, user

    @property
    def template_hash(self) -> str:
        """Content hash of the template bodies and their rendering flags.

        Feeds `params_hash`. Twelve hex characters — 48 bits, which is far
        more than enough to separate the handful of templates a project like
        this will ever have, while staying readable in a filename.
        """
        h = hashlib.sha256()
        for part in (self.system, self.user, str(self.include_visible_tests)):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:12]


_SYSTEM_DIRECT = (
    "You are an expert Python programmer. You write correct, complete, "
    "self-contained solutions.\n\n"
    "Respond with exactly one Python code block and nothing else. No "
    "explanation before it, no commentary after it.\n\n"
    "The code block must contain the complete module: every import it needs "
    "and every function it defines. It is written to a file and imported "
    "directly, so it must run top to bottom without errors and must not read "
    "from stdin, print at import time, or call sys.exit."
)

_USER_DIRECT = """Write a Python solution to the following problem.

{prompt}{tests_block}

Respond with one ```python code block containing the complete solution."""

_SYSTEM_REPAIR = (
    "You are an expert Python programmer fixing a failing solution.\n\n"
    "Respond with exactly one Python code block and nothing else, containing "
    "the complete corrected module — not a diff, not a fragment, and not only "
    "the changed function.\n\n"
    "Read the reported failure before rewriting. If the previous attempt is "
    "close, make the smallest change that fixes it; if its approach cannot "
    "work, replace it."
)

_USER_REPAIR = """The following problem was solved incorrectly.

{prompt}{tests_block}{prior_block}

Respond with one ```python code block containing the complete corrected solution."""


TEMPLATES: dict[str, PromptTemplate] = {
    "direct_v1": PromptTemplate(
        template_id="direct_v1",
        system=_SYSTEM_DIRECT,
        user=_USER_DIRECT,
    ),
    # Ablation arm: identical instructions, no visible tests shown. Isolates
    # how much of the small model's accuracy comes from seeing the tests.
    "direct_notests_v1": PromptTemplate(
        template_id="direct_notests_v1",
        system=_SYSTEM_DIRECT,
        user=_USER_DIRECT,
        include_visible_tests=False,
    ),
    "repair_v1": PromptTemplate(
        template_id="repair_v1",
        system=_SYSTEM_REPAIR,
        user=_USER_REPAIR,
    ),
}


def get_template(template_id: str) -> PromptTemplate:
    """Look up a template, failing loudly on a typo.

    A missing template must not fall back to a default — that would silently
    change what was generated while leaving the recorded `template_id` intact.
    """
    try:
        return TEMPLATES[template_id]
    except KeyError:
        raise KeyError(
            f"unknown prompt template {template_id!r}; "
            f"known templates: {sorted(TEMPLATES)}"
        ) from None
