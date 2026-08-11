"""The store's three contracts: append-only, write-then-seal, checksummed."""

from __future__ import annotations

import json

import pytest

from orchestrator.workers.errors import StoreError, UnsealedRunError
from orchestrator.workers.generation import Generation
from orchestrator.workers.store import (
    RolloutStore, list_runs, read_generations, read_manifest, read_rows,
)

RUN_ID = "2026-08-11-abc1234-def456"


def gen(task_id="t0", arm="direct_small", seed=0, finish="stop", **kwargs):
    return Generation(
        run_id=RUN_ID, task_id=task_id, arm=arm, seed=seed,
        params_hash="c0ff9e5c88e4", text="```python\nx=1\n```", code="x = 1",
        model_id="mock-small", prefill_tokens=100, decode_tokens=50,
        finish_reason=finish, extract_strategy="fenced_parsed", code_parses=True,
        **kwargs,
    )


def test_rows_survive_a_write_read_round_trip(runs_root):
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append_many([gen(task_id=f"t{i}") for i in range(5)])
    store.seal()
    rows = list(read_generations(runs_root, RUN_ID))
    assert [r.task_id for r in rows] == [f"t{i}" for i in range(5)]


def test_readers_skip_unsealed_runs(runs_root):
    """An interrupted sweep read as a complete one produces a real number
    computed over a fraction of the corpus."""
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen())
    store.close()
    with pytest.raises(UnsealedRunError):
        list(read_rows(runs_root, RUN_ID))
    assert list_runs(runs_root) == []
    assert list_runs(runs_root, sealed_only=False) == [RUN_ID]


def test_resume_may_read_an_unsealed_run(runs_root):
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen())
    store.close()
    assert len(list(read_rows(runs_root, RUN_ID, allow_unsealed=True))) == 1


def test_a_sealed_run_cannot_be_appended_to(runs_root):
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen())
    store.seal()
    with pytest.raises(StoreError, match="sealed"):
        RolloutStore(runs_root, RUN_ID).open()


def test_a_run_cannot_be_sealed_twice(runs_root):
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen())
    store.seal()
    with pytest.raises(StoreError, match="already sealed"):
        RolloutStore(runs_root, RUN_ID).seal()


def test_manifest_counts_every_part_not_just_this_process(runs_root):
    """A resumed run has rows an earlier process wrote and this one never saw."""
    first = RolloutStore(runs_root, RUN_ID).open()
    first.append_many([gen(task_id=f"a{i}") for i in range(3)])
    first.close()
    second = RolloutStore(runs_root, RUN_ID).open()
    second.append_many([gen(task_id=f"b{i}") for i in range(2)])
    manifest = second.seal()
    assert manifest["n_rows"] == 5
    assert len(manifest["files"]) == 2


def test_manifest_records_a_checksum_per_part(runs_root):
    """R4's golden-run test needs to detect a store that changed underneath."""
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen())
    manifest = store.seal()
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])
    assert all(f["rows"] > 0 for f in manifest["files"])


def test_manifest_reports_the_truncation_rate(runs_root):
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append_many([gen(task_id=f"t{i}") for i in range(3)])
    store.append(gen(task_id="t3", finish="length"))
    manifest = store.seal()
    assert manifest["truncation_rate"] == pytest.approx(0.25)
    assert manifest["counts"]["by_finish_reason"] == {"length": 1, "stop": 3}


def test_manifest_records_one_params_hash_per_arm(runs_root):
    """More than one means the run holds two experiments."""
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append_many([gen(task_id=f"t{i}") for i in range(3)])
    manifest = store.seal()
    assert manifest["params_hash_by_arm"] == {"direct_small": ["c0ff9e5c88e4"]}


def test_a_dirty_worktree_is_not_publishable(runs_root, monkeypatch):
    monkeypatch.setattr("orchestrator.workers.store.is_dirty", lambda *a, **k: True)
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen())
    assert store.seal()["publishable"] is False


def test_a_dirty_run_id_is_not_publishable(runs_root):
    store = RolloutStore(runs_root, RUN_ID + "-dirty").open()
    store.append(gen())
    assert store.seal()["publishable"] is False


def test_a_torn_final_line_is_skipped_not_fatal(runs_root):
    """A sweep killed mid-write leaves a partial line; resume regenerates
    that cell, which is the correct outcome."""
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append_many([gen(task_id=f"t{i}") for i in range(3)])
    store.close()
    part = store.part_files()[0]
    with part.open("a", encoding="utf-8", newline="") as fh:
        fh.write('{"task_id": "torn", "arm": ')
    assert len(list(read_rows(runs_root, RUN_ID, allow_unsealed=True))) == 3


def test_rows_carry_null_grading_fields(runs_root):
    """Emitting the keys means an ungraded read has the same columns as a
    graded one, so nothing downstream special-cases it."""
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen())
    store.seal()
    row = next(iter(read_rows(runs_root, RUN_ID)))
    for field in ("visible_passed", "hidden_passed", "error_class", "hack_flags"):
        assert field in row and row[field] is None


def test_line_endings_do_not_depend_on_the_platform(runs_root):
    """A part file's sha256 must be the same on every OS."""
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen())
    store.close()
    assert b"\r\n" not in store.part_files()[0].read_bytes()


def test_reading_a_missing_run_raises(runs_root):
    with pytest.raises(StoreError, match="no such run"):
        list(read_rows(runs_root, "2026-01-01-0000000-000000"))


def test_reading_an_unsealed_manifest_raises(runs_root):
    RolloutStore(runs_root, RUN_ID).open().close()
    with pytest.raises(UnsealedRunError):
        read_manifest(runs_root, RUN_ID)


def test_config_is_written_when_the_run_opens(runs_root):
    store = RolloutStore(runs_root, RUN_ID).open(config={"arms": ["direct_small"]})
    store.close()
    written = json.loads((store.dir / "_CONFIG.json").read_text(encoding="utf-8"))
    assert written == {"arms": ["direct_small"]}
