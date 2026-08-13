"""Loading R2's task manifest and R4's split manifest.

Every check here is cheap and runs before the first token, because a malformed
corpus discovered at hour nine of a sweep costs GPU time.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.workers.corpus import (
    CorpusError, build_corpus, load_splits, load_tasks,
)

from conftest import write_corpus


def test_tasks_load_from_a_file(tmp_path):
    tasks = load_tasks(write_corpus(tmp_path / "t.jsonl", n=5))
    assert len(tasks) == 5
    assert tasks[0].entrypoint == "add_0"
    assert tasks[0].metadata["visible_tests"]


def test_tasks_load_from_a_directory(tmp_path):
    write_corpus(tmp_path / "tasks" / "a.jsonl", n=3)
    tasks = load_tasks(tmp_path / "tasks")
    assert len(tasks) == 3


def test_a_duplicate_task_id_is_refused(tmp_path):
    """A duplicate makes the resume key ambiguous, so resume either skips real
    work or overwrites it."""
    path = tmp_path / "t.jsonl"
    record = {"task_id": "dup", "prompt": "p"}
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(CorpusError, match="duplicate task_id"):
        load_tasks(path)


def test_a_malformed_line_reports_its_line_number(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"task_id": "a", "prompt": "p"}\nnot json\n')
    with pytest.raises(CorpusError, match=":2:"):
        load_tasks(path)


def test_a_missing_task_id_is_refused(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"prompt": "p"}\n')
    with pytest.raises(CorpusError, match="no task_id"):
        load_tasks(path)


def test_an_empty_prompt_is_refused(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"task_id": "a", "prompt": "   "}\n')
    with pytest.raises(CorpusError, match="no prompt"):
        load_tasks(path)


def test_a_missing_manifest_names_its_owner(tmp_path):
    with pytest.raises(CorpusError, match="R2 owns"):
        load_tasks(tmp_path / "absent.jsonl")


@pytest.mark.parametrize("id_key", ["task_id", "id", "name"])
def test_alternative_id_spellings_are_accepted(tmp_path, id_key):
    """The corpus is not R1's to define."""
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({id_key: "a", "prompt": "p"}) + "\n")
    assert load_tasks(path)[0].task_id == "a"


def test_tests_given_as_a_list_are_joined(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({
        "task_id": "a", "prompt": "p", "test_list": ["assert f(1)", "assert f(2)"],
    }) + "\n")
    assert load_tasks(path)[0].tests == "assert f(1)\nassert f(2)"


def test_require_tests_is_opt_in(tmp_path):
    """R1 does not need tests to generate; a manifest whose hidden tests are
    not attached yet must not block a sweep."""
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"task_id": "a", "prompt": "p"}) + "\n")
    assert load_tasks(path)
    with pytest.raises(CorpusError, match="no tests"):
        load_tasks(path, require_tests=True)


@pytest.mark.parametrize("shape", ["flat", "by_split", "nested", "task_ids"])
def test_split_manifest_shapes(tmp_path, shape):
    path = tmp_path / "splits.json"
    data = {
        "flat": {"a": "train", "b": "test"},
        "by_split": {"train": ["a"], "test": ["b"]},
        "nested": {"splits": {"a": "train", "b": "test"}},
        "task_ids": {"task_ids": {"a": "train", "b": "test"}},
    }[shape]
    path.write_text(json.dumps(data))
    assert load_splits(path) == {"a": "train", "b": "test"}


def test_sibling_metadata_is_not_read_as_task_ids(tmp_path):
    """R4's manifest puts the mapping under `task_ids` and sits it beside
    string metadata. Reading the top level flat turns `corpus_hash`, `name`
    and `salt` into three task ids and leaves every real task unassigned."""
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({
        "corpus_hash": "fcc0a6fd", "name": "pilot_200", "salt": "pilot-2026-08-13",
        "n_tasks": 2, "task_ids": {"a": "train", "b": "test"},
    }))
    assert load_splits(path) == {"a": "train", "b": "test"}


def test_a_manifest_that_assigns_nothing_is_an_error(tmp_path):
    """Silently unassigning every task makes `--include-splits` inert, so the
    test-split fence stops fencing without anyone being told."""
    corpus_path = write_corpus(tmp_path / "t.jsonl", n=3)
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"not/a/real/id": "train"}))
    with pytest.raises(CorpusError, match="assigns no task"):
        build_corpus(corpus_path, splits_path)


def test_splits_attach_to_tasks(tmp_path):
    corpus_path = write_corpus(tmp_path / "t.jsonl", n=6)
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"train": ["mbpp/0", "mbpp/1"]}))
    corpus = build_corpus(corpus_path, splits_path)
    assert corpus.split_of("mbpp/0") == "train"


def test_the_split_manifest_wins_over_a_split_on_the_record(tmp_path):
    corpus_path = write_corpus(tmp_path / "t.jsonl", n=3, splits=("val",))
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"mbpp/0": "train"}))
    corpus = build_corpus(corpus_path, splits_path)
    assert corpus.split_of("mbpp/0") == "train"
    assert corpus.split_of("mbpp/1") == "val"


def test_filtering_to_a_split(tmp_path):
    corpus = build_corpus(
        write_corpus(tmp_path / "t.jsonl", n=9), include_splits=["train"],
    )
    assert {corpus.split_of(t.task_id) for t in corpus.tasks} == {"train"}


def test_filtering_to_an_absent_split_raises(tmp_path):
    with pytest.raises(CorpusError, match="no tasks left"):
        build_corpus(write_corpus(tmp_path / "t.jsonl", n=3),
                     include_splits=["nonexistent"])


def test_limit_takes_manifest_order_not_a_random_sample(tmp_path):
    """A random pilot subset would differ between two runs of the same pilot."""
    corpus = build_corpus(write_corpus(tmp_path / "t.jsonl", n=20), limit=5)
    assert [t.task_id for t in corpus.tasks] == [f"mbpp/{i}" for i in range(5)]


def test_counts_report_per_split_totals(tmp_path):
    corpus = build_corpus(write_corpus(tmp_path / "t.jsonl", n=9))
    assert corpus.counts() == {"test": 3, "train": 3, "val": 3}
