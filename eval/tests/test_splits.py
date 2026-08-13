"""Splits must survive the corpus growing — that is the whole design.

The central test is `test_growing_the_corpus_does_not_move_existing_tasks`.
Under a shuffle-based split it fails, silently and unrecoverably: a task that
was visible during the pilot lands in `test` for the real run, and the file
still looks fine.
"""

from __future__ import annotations

import json

import pytest

from eval.splits import (
    DEFAULT_RATIOS,
    SPLITS,
    SplitError,
    SplitManifest,
    apply_to_tasks,
    assign,
    build,
    corpus_hash,
    read_corpus,
    verify,
)

SALT = "pilot-2026-08"


def _tasks(n: int, *, prefix: str = "t", prompt: str = "p"):
    return [
        {"task_id": f"{prefix}/{i:05d}", "prompt": f"{prompt}{i}",
         "visible_tests": "assert f(1)", "hidden_tests": "assert f(2)"}
        for i in range(n)
    ]


# --- the property the design exists for ------------------------------------

def test_growing_the_corpus_does_not_move_existing_tasks():
    """Add 800 tasks to a 200-task pilot; the original 200 keep their splits.

    A shuffle-based implementation reassigns all of them, putting tasks that
    were visible during the pilot into the frozen test set.
    """
    pilot = build(_tasks(200), name="pilot", salt=SALT)
    full = build(_tasks(1000), name="full", salt=SALT)
    for task_id, split in pilot.task_ids.items():
        assert full.task_ids[task_id] == split, task_id


def test_assignment_is_order_independent():
    forward = build(_tasks(300), name="a", salt=SALT)
    backward = build(list(reversed(_tasks(300))), name="a", salt=SALT)
    assert forward.task_ids == backward.task_ids


def test_removing_tasks_does_not_move_the_survivors():
    full = build(_tasks(500), name="a", salt=SALT)
    subset = build(_tasks(500)[::2], name="a", salt=SALT)
    for task_id, split in subset.task_ids.items():
        assert full.task_ids[task_id] == split


def test_assignment_is_stable_across_processes():
    """sha256, not Python's salted `hash()` — which differs per interpreter run
    and would make the split irreproducible in the least visible way possible."""
    assert assign("t/00042", salt=SALT) == assign("t/00042", salt=SALT)
    assert assign("t/00042", salt=SALT) in SPLITS


def test_a_different_salt_reshuffles():
    a = build(_tasks(400), name="a", salt="one")
    b = build(_tasks(400), name="a", salt="two")
    moved = sum(1 for k, v in a.task_ids.items() if b.task_ids[k] != v)
    assert moved > 100, "a new salt must produce a genuinely different split"


# --- proportions -----------------------------------------------------------

def test_ratios_are_approximately_respected():
    manifest = build(_tasks(4000), name="a", salt=SALT)
    for split, ratio in zip(SPLITS, DEFAULT_RATIOS):
        observed = manifest.counts[split] / manifest.n_tasks
        assert abs(observed - ratio) < 0.02, f"{split}: {observed:.3f}"


def test_counts_sum_to_the_corpus_size():
    manifest = build(_tasks(777), name="a", salt=SALT)
    assert sum(manifest.counts.values()) == manifest.n_tasks == 777


def test_custom_ratios_are_honoured():
    manifest = build(_tasks(4000), name="a", salt=SALT, ratios=(0.8, 0.1, 0.1))
    assert abs(manifest.counts["train"] / 4000 - 0.8) < 0.02


def test_ratios_must_sum_to_one():
    with pytest.raises(SplitError, match="sum to 1.0"):
        build(_tasks(10), name="a", salt=SALT, ratios=(0.5, 0.2, 0.2))


def test_empty_salt_is_refused():
    """The salt is the split's identity; an empty one makes two different
    splits indistinguishable."""
    with pytest.raises(SplitError, match="salt"):
        build(_tasks(10), name="a", salt="  ")


def test_duplicate_task_ids_are_refused():
    tasks = _tasks(10) + _tasks(3)
    with pytest.raises(SplitError, match="duplicate task_ids"):
        build(tasks, name="a", salt=SALT)


# --- verification ----------------------------------------------------------

def test_verify_passes_on_the_corpus_it_was_built_from():
    tasks = _tasks(200)
    assert verify(build(tasks, name="a", salt=SALT), tasks) == []


def test_verify_catches_a_mutated_corpus():
    """R2 regenerating data/tasks/ in place: same ids, different content. The
    tasks behind these ids are no longer the tasks that were split."""
    manifest = build(_tasks(200), name="a", salt=SALT)
    problems = verify(manifest, _tasks(200, prompt="CHANGED"))
    assert any("corpus content hash changed" in p for p in problems)


def test_verify_catches_a_hand_edited_manifest():
    """The edit someone makes at 2am in week 9: move one stubborn task out of
    test. A stored-only mapping could not detect this."""
    tasks = _tasks(200)
    manifest = build(tasks, name="a", salt=SALT)
    victim = next(k for k, v in manifest.task_ids.items() if v == "test")
    tampered = SplitManifest(
        name=manifest.name, salt=manifest.salt, ratios=manifest.ratios,
        corpus_hash=manifest.corpus_hash, counts=manifest.counts,
        n_tasks=manifest.n_tasks,
        task_ids={**manifest.task_ids, victim: "train"},
    )
    problems = verify(tampered, tasks)
    assert any("edited by hand" in p for p in problems)


def test_verify_catches_tasks_missing_from_the_manifest():
    manifest = build(_tasks(100), name="a", salt=SALT)
    problems = verify(manifest, _tasks(150), require_same_corpus=False)
    assert any("absent from the manifest" in p for p in problems)


def test_verify_reports_every_problem_not_just_the_first():
    manifest = build(_tasks(100), name="a", salt=SALT)
    problems = verify(manifest, _tasks(150, prompt="CHANGED"))
    assert len(problems) >= 2


# --- round trip ------------------------------------------------------------

def test_manifest_round_trips_through_disk(tmp_path):
    manifest = build(_tasks(120), name="pilot", salt=SALT)
    path = manifest.write(tmp_path / "splits.json")
    assert SplitManifest.read(path).as_dict() == manifest.as_dict()


def test_manifest_file_is_deterministic(tmp_path):
    """Committed artifacts must not produce a diff when regenerated."""
    a = build(_tasks(120), name="pilot", salt=SALT).write(tmp_path / "a.json")
    b = build(_tasks(120), name="pilot", salt=SALT).write(tmp_path / "b.json")
    assert a.read_text() == b.read_text()


def test_split_of_works_for_a_task_added_after_freezing():
    """Recomputation, not lookup: a task absent from the stored map still gets
    the right answer when the corpus grows."""
    manifest = build(_tasks(50), name="a", salt=SALT)
    assert manifest.split_of("t/99999") in SPLITS
    assert "t/99999" not in manifest.task_ids


def test_apply_to_tasks_attaches_the_split_field():
    tasks = _tasks(30)
    manifest = build(tasks, name="a", salt=SALT)
    tagged = apply_to_tasks(manifest, tasks)
    assert all(t["split"] in SPLITS for t in tagged)
    assert all(t["split"] == manifest.task_ids[t["task_id"]] for t in tagged)


def test_corpus_hash_ignores_ordering():
    tasks = _tasks(40)
    assert corpus_hash(tasks) == corpus_hash(list(reversed(tasks)))


def test_corpus_hash_notices_a_changed_hidden_test():
    """Hidden tests define the label. A change there changes every outcome."""
    tasks = _tasks(10)
    mutated = [dict(t) for t in tasks]
    mutated[0]["hidden_tests"] = "assert f(3)"
    assert corpus_hash(tasks) != corpus_hash(mutated)


def test_read_corpus_rejects_an_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SplitError, match="no tasks"):
        read_corpus(empty)


def test_read_corpus_reads_jsonl(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in _tasks(5)) + "\n")
    assert len(read_corpus(path)) == 5


def test_ratio_tolerance_shrinks_as_the_corpus_grows():
    """A fixed tolerance would wave through a broken split at n=4000 while
    failing a healthy one at n=200 — which is what it did the first time."""
    from eval.splits import ratio_tolerance

    assert ratio_tolerance(0.6, 200) > ratio_tolerance(0.6, 4000)
    assert ratio_tolerance(0.6, 200) > 0.09, "n=200 is genuinely noisy"
    assert ratio_tolerance(0.6, 100_000) < 0.011, "a huge corpus must be tight"


def test_verify_still_catches_a_genuinely_wrong_ratio():
    """Loosening the tolerance must not disarm the check."""
    tasks = _tasks(4000)
    manifest = build(tasks, name="a", salt=SALT, ratios=(0.9, 0.05, 0.05))
    lying = SplitManifest(
        name=manifest.name, salt=manifest.salt, ratios=DEFAULT_RATIOS,
        corpus_hash=manifest.corpus_hash, counts=manifest.counts,
        n_tasks=manifest.n_tasks, task_ids=manifest.task_ids,
    )
    assert any("standard errors" in p for p in verify(lying, tasks))
