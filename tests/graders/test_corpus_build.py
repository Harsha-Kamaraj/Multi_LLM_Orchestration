"""Tests run entirely offline against synthetic EvalPlus-shaped records —
matching the real schema (`task_id`, `prompt`, `entry_point`,
`canonical_solution`, `base_input`, `plus_input`, `atol`) confirmed by
inspecting the installed `evalplus` package, but never touching the network
or the real dataset. `build_pilot` itself (which does fetch for real) is
exercised separately, by hand, when actually building the corpus.
"""

from __future__ import annotations

from orchestrator.graders.corpus_build import (
    contamination_filter,
    to_task_record,
)
from orchestrator.graders.pytest_grader import PytestGrader
from orchestrator.types import Task
from schemas import validate_task

ADD_PROBLEM = {
    "task_id": "Test/1",
    "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
    "entry_point": "add",
    "canonical_solution": "    return a + b\n",
    "base_input": [[1, 2], [0, 0]],
    "plus_input": [[5, 5], [-1, 1], [100, -100]],
    "atol": 0,
}

BROKEN_PROBLEM = {
    "task_id": "Test/2",
    "prompt": "def broken(a):\n    pass\n",
    "entry_point": "does_not_exist",
    "canonical_solution": "    return a\n",
    "base_input": [[1]],
    "plus_input": [],
    "atol": 0,
}

FLOAT_PROBLEM = {
    "task_id": "Test/3",
    "prompt": "def half(a):\n    \"\"\"Halve a number.\"\"\"\n",
    "entry_point": "half",
    "canonical_solution": "    return a / 2\n",
    "base_input": [[10.0]],
    "plus_input": [[3.0]],
    "atol": 0.001,
}


def test_to_task_record_shape():
    record = to_task_record(ADD_PROBLEM, "testset")
    assert record["task_id"] == "Test/1"
    assert record["entrypoint"] == "add"
    assert record["metadata"]["n_visible_cases"] == 2
    assert record["metadata"]["n_hidden_cases"] == 5  # base + plus
    assert "candidate(*[1, 2])" in record["visible_tests"]
    assert "candidate(*[100, -100])" in record["tests"]
    assert "candidate(*[100, -100])" not in record["visible_tests"]  # hidden-only case


def test_to_task_record_returns_none_for_unconvertible():
    assert to_task_record(BROKEN_PROBLEM, "testset") is None


def test_to_task_record_conforms_to_frozen_schema():
    """R4's schemas/task.schema.json is the authority, not this module's own
    idea of what a task looks like — validate against the real thing."""
    record = to_task_record(ADD_PROBLEM, "testset")
    validate_task(record)  # raises ValidationError on drift
    assert record["dataset"] == "testset"
    assert record["hidden_tests"] == record["tests"]  # bridge field, same content


def test_generated_tests_are_actually_gradeable():
    """Round-trip: the generated visible/hidden pytest modules must be valid
    Python that PytestGrader can execute — this is what actually matters,
    more than any property of the generator's output as text."""
    record = to_task_record(ADD_PROBLEM, "testset")
    task = Task(
        task_id=record["task_id"], prompt=record["prompt"],
        entrypoint=record["entrypoint"], tests=record["tests"],
        metadata={"visible_tests": record["visible_tests"]},
    )
    grader = PytestGrader(backend="subprocess")

    correct = grader.grade(task, "def add(a, b):\n    return a + b\n")
    assert correct.visible.all_passed
    assert correct.hidden.all_passed
    assert correct.solved

    wrong = grader.grade(task, "def add(a, b):\n    return a - b\n")
    assert not wrong.solved


def test_float_tolerance_is_respected():
    record = to_task_record(FLOAT_PROBLEM, "testset")
    task = Task(
        task_id=record["task_id"], prompt=record["prompt"],
        entrypoint=record["entrypoint"], tests=record["tests"],
        metadata={"visible_tests": record["visible_tests"]},
    )
    grader = PytestGrader(backend="subprocess")
    # Off by less than atol (0.001) — must still pass.
    grade = grader.grade(task, "def half(a):\n    return a / 2 + 0.0001\n")
    assert grade.hidden.all_passed


def test_contamination_filter_drops_duplicates():
    records = [
        {"task_id": "a", "prompt": "write a function that adds two numbers together please"},
        {"task_id": "b", "prompt": "write a function that adds two numbers together please"},
        {"task_id": "c", "prompt": "write a different function entirely, subtracting instead"},
    ]
    kept, filtered = contamination_filter(records)
    assert [r["task_id"] for r in kept] == ["a", "c"]
    assert filtered == [{"task_id": "b", "reason": "duplicate_prompt_of:a"}]


def test_contamination_filter_drops_degenerate_prompts():
    records = [{"task_id": "a", "prompt": "too short"}]
    kept, filtered = contamination_filter(records)
    assert kept == []
    assert filtered == [{"task_id": "a", "reason": "degenerate_prompt"}]
