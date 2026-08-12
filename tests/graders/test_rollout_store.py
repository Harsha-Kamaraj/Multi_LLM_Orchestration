from __future__ import annotations

import pytest

from orchestrator.graders.errors import GraderError
from orchestrator.graders.rollout_store import (
    RolloutStore,
    list_graded_runs,
    read_manifest,
    read_rows,
)


def _row(rollout_id: str, **overrides) -> dict:
    base = {
        "rollout_id": rollout_id,
        "task_id": "t1",
        "visible_passed": 1,
        "visible_total": 1,
        "hidden_passed": 1,
        "hidden_total": 1,
        "error_class": "none",
        "hack_flags": [],
        "grade_duration_s": 0.1,
    }
    base.update(overrides)
    return base


def test_append_and_seal_roundtrip(tmp_path):
    with RolloutStore(tmp_path, "run-a").open() as store:
        store.append(_row("r1"))
        store.append(_row("r2", hidden_passed=0))
    manifest = RolloutStore(tmp_path, "run-a").seal()
    assert manifest["n_rows"] == 2
    assert manifest["solved_count"] == 1

    rows = list(read_rows(tmp_path, "run-a"))
    assert {r["rollout_id"] for r in rows} == {"r1", "r2"}


def test_unsealed_run_is_unreadable_by_default(tmp_path):
    with RolloutStore(tmp_path, "run-b").open() as store:
        store.append(_row("r1"))
    with pytest.raises(GraderError):
        list(read_rows(tmp_path, "run-b"))
    assert list(read_rows(tmp_path, "run-b", allow_unsealed=True))


def test_sealed_run_refuses_further_appends(tmp_path):
    with RolloutStore(tmp_path, "run-c").open() as store:
        store.append(_row("r1"))
    RolloutStore(tmp_path, "run-c").seal()

    with pytest.raises(GraderError):
        RolloutStore(tmp_path, "run-c").open()


def test_seal_is_not_repeatable(tmp_path):
    with RolloutStore(tmp_path, "run-d").open() as store:
        store.append(_row("r1"))
    store2 = RolloutStore(tmp_path, "run-d")
    store2.seal()
    with pytest.raises(GraderError):
        store2.seal()


def test_append_requires_rollout_id(tmp_path):
    with RolloutStore(tmp_path, "run-e").open() as store:
        with pytest.raises(GraderError):
            store.append({"task_id": "t1"})


def test_list_and_read_manifest(tmp_path):
    with RolloutStore(tmp_path, "run-f").open() as store:
        store.append(_row("r1"))
    assert list_graded_runs(tmp_path) == []  # not sealed yet
    RolloutStore(tmp_path, "run-f").seal()
    assert list_graded_runs(tmp_path) == ["run-f"]
    manifest = read_manifest(tmp_path, "run-f")
    assert manifest["run_id"] == "run-f"


def test_read_manifest_missing_raises(tmp_path):
    with pytest.raises(GraderError):
        read_manifest(tmp_path, "no-such-run")
