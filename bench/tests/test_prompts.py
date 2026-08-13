"""Prompt rendering, and the guard that keeps labels out of prompts."""

from __future__ import annotations

import pytest

from orchestrator.types import Task
from orchestrator.workers.prompts import (
    TEMPLATES, PromptContext, PromptTemplate, get_template,
)


def _task(**metadata) -> Task:
    return Task(
        task_id="t1", prompt="Add two numbers.", tests="hidden and visible mixed",
        entrypoint="add", metadata=metadata,
    )


def test_visible_tests_are_rendered():
    ctx = PromptContext.from_task(_task(visible_tests="assert add(1, 2) == 3"))
    _system, user = get_template("direct_v1").render(ctx)
    assert "assert add(1, 2) == 3" in user


def test_task_tests_are_never_rendered():
    """`Task.tests` is the full suite including hidden cases. Falling back to
    it when no visible split exists would put labels in the prompt."""
    ctx = PromptContext.from_task(_task())
    _system, user = get_template("direct_v1").render(ctx)
    assert "hidden and visible mixed" not in user


def test_context_cannot_reach_the_full_suite():
    assert not hasattr(PromptContext.from_task(_task()), "tests")


@pytest.mark.parametrize("key", [
    "hidden_tests", "hidden_passed", "canonical_solution", "reference_solution",
    "gold", "answer",
])
def test_label_bearing_metadata_is_refused(key):
    with pytest.raises(ValueError, match="label"):
        PromptContext.from_task(_task(**{key: "boom"}))


@pytest.mark.parametrize("key", ["n_solutions_found", "source", "difficulty_bucket"])
def test_innocent_metadata_is_allowed(key):
    """A guard that trips on harmless keys blocks a sweep for no reason."""
    assert PromptContext.from_task(_task(**{key: 3})) is not None


@pytest.mark.parametrize("key", ["n_hidden_cases", "n_visible_cases"])
def test_scalar_counts_about_the_hidden_suite_are_allowed(key):
    """R2's frozen manifest carries `n_hidden_cases`. It names the hidden suite
    without containing one, and a count is not a label — refusing it blocks the
    pilot sweep on a key no template can reach."""
    assert PromptContext.from_task(_task(**{key: 1006})) is not None


def test_hidden_key_holding_content_is_still_refused():
    """The scalar exemption must not become a way to smuggle a suite in."""
    with pytest.raises(ValueError, match="label"):
        PromptContext.from_task(_task(hidden_tests="assert add(1, 2) == 3"))


def test_notests_template_omits_the_tests():
    ctx = PromptContext.from_task(_task(visible_tests="assert add(1, 2) == 3"))
    _system, user = get_template("direct_notests_v1").render(ctx)
    assert "assert add(1, 2) == 3" not in user


def test_repair_template_carries_the_prior_attempt_and_feedback():
    ctx = PromptContext.from_task(
        _task(), prior_code="def add(a, b): return a - b",
        prior_feedback="test_add failed: expected 3, got -1",
    )
    _system, user = get_template("repair_v1").render(ctx)
    assert "a - b" in user and "expected 3, got -1" in user


def test_template_hash_follows_the_text():
    original = TEMPLATES["direct_v1"]
    edited = PromptTemplate(
        template_id=original.template_id, system=original.system,
        user=original.user + " Be brief.",
    )
    assert edited.template_hash != original.template_hash


def test_template_hash_ignores_the_label():
    original = TEMPLATES["direct_v1"]
    renamed = PromptTemplate(
        template_id="something_else", system=original.system, user=original.user,
    )
    assert renamed.template_hash == original.template_hash


def test_rendering_flags_are_hashed():
    """`include_visible_tests` changes the rendered text and must be covered."""
    assert (
        TEMPLATES["direct_v1"].template_hash
        != TEMPLATES["direct_notests_v1"].template_hash
    )


def test_unknown_template_raises_rather_than_defaulting():
    """A fallback would generate under different instructions than the ones
    recorded on the row."""
    with pytest.raises(KeyError, match="unknown prompt template"):
        get_template("nope_v9")


def test_braces_in_a_prompt_survive_rendering():
    task = Task(task_id="t", prompt="Return {'a': 1} as a dict.", tests="", entrypoint="f")
    _system, user = get_template("direct_v1").render(PromptContext.from_task(task))
    assert "{'a': 1}" in user
