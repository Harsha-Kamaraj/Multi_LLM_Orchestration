"""The reader, and the four things it refuses.

Two of these tests are interop tests rather than unit tests: one builds a run
with R1's own `RolloutStore` and seals it, and one reads that same run through
both the Parquet and the JSONL path and asserts they agree row for row. Those
are the tests that fail when the contract drifts, which is the only warning R3
gets before a store stops meaning what this code assumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator.policy import store
from orchestrator.policy.errors import (
    SchemaVersionError, SplitError, StoreReadError, UngradedRunError,
)
from orchestrator.workers.generation import Generation
from orchestrator.workers.store import RolloutStore

RUN_ID = "2026-08-12-abc1234-def567"


def make_generation(task: int, arm: str = "direct_small", seed: int = 0,
                    split: str = "train", **overrides: Any) -> Generation:
    fields: dict[str, Any] = dict(
        run_id=RUN_ID,
        task_id=f"mbpp/{task}",
        arm=arm,
        seed=seed,
        params_hash="p" * 12,
        text=f"```python\ndef f{task}():\n    return {task}\n```",
        code=f"def f{task}():\n    return {task}\n",
        model_id="mock-small" if arm == "direct_small" else "mock-large",
        prefill_tokens=100 + task,
        decode_tokens=40 + task,
        wall_ms=300.0 + task,
        finish_reason="stop",
        backend="mock",
        extract_strategy="fenced",
        code_parses=True,
        split=split,
        dataset="mbpp+",
        code_version="abc1234",
    )
    fields.update(overrides)
    return Generation(**fields)


def graded(gen: Generation, *, hidden_passed: int = 8, hidden_total: int = 8,
           visible_passed: int = 3, visible_total: int = 3) -> dict[str, Any]:
    """R1's row with R2's columns filled — what a training example looks like."""
    row = gen.to_row()
    row.update({
        "visible_passed": visible_passed,
        "visible_total": visible_total,
        "hidden_passed": hidden_passed,
        "hidden_total": hidden_total,
        "error_class": None,
        "hack_flags": [],
        "grade_duration_s": 0.4,
    })
    return row


def write_run(root: Path, rows: list[dict[str, Any]], *, run_id: str = RUN_ID,
              sealed: bool = True, publishable: bool = True) -> Path:
    """Write a run directory by hand, so a test can control every byte.

    Phase 2 promotes this into a real fixture generator. Here it exists so the
    refusal paths — unsealed, damaged, mixed-version — can be produced exactly.
    """
    directory = root / run_id
    rows_dir = directory / "generations"
    rows_dir.mkdir(parents=True, exist_ok=True)
    part = rows_dir / "part-000001-0000000000-aaaaaa.jsonl"
    with part.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    if sealed:
        (directory / "_MANIFEST.json").write_text(json.dumps({
            "run_id": run_id,
            "schema_version": 1,
            "n_rows": len(rows),
            "publishable": publishable,
        }, indent=2), encoding="utf-8")
    return directory


def write_cost(root: Path, rows: list[dict[str, Any]], fingerprint: str,
               *, run_id: str = RUN_ID) -> None:
    directory = root / run_id / "cost"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{fingerprint}.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({
                "rollout_id": row["rollout_id"],
                "task_id": row["task_id"],
                "arm": row["arm"],
                "seed": row["seed"],
                "model_id": row["model_id"],
                "gpu_seconds": 0.5,
                "imputed_latency_s": 1.25,
                "usd": 0.0004,
            }) + "\n")


@pytest.fixture
def graded_run(tmp_path: Path) -> Path:
    rows = [
        graded(make_generation(i, arm=arm, seed=seed),
               hidden_passed=8 if i % 2 == 0 else 3)
        for i in range(4)
        for arm in ("direct_small", "direct_large")
        for seed in (0, 1)
    ]
    write_run(tmp_path, rows)
    return tmp_path


# -- interop with R1 ---------------------------------------------------------


def test_reads_a_run_written_and_sealed_by_r1(tmp_path: Path):
    """The drift canary. If R1's layout changes, this is where R3 finds out."""
    with RolloutStore(tmp_path, RUN_ID).open(config={"note": "test"}) as rollouts:
        for i in range(6):
            rollouts.append(make_generation(i))
        rollouts.seal()

    data = store.load_rollouts(tmp_path, RUN_ID, require_grades=False)
    assert len(data) == 6
    assert data.run_id == RUN_ID
    assert data.arms == ("direct_small",)


def test_parquet_and_jsonl_read_the_same_run_identically(tmp_path: Path):
    """Without this, a result reproduces on one machine and not another."""
    pytest.importorskip("pyarrow")
    with RolloutStore(tmp_path, RUN_ID).open() as rollouts:
        for i in range(6):
            rollouts.append(make_generation(i))
        manifest = rollouts.seal()
    assert manifest["parquet"], "R1 did not write a Parquet view to compare"

    via_parquet = store.load_rollouts(tmp_path, RUN_ID, require_grades=False,
                                      prefer_parquet=True)
    via_jsonl = store.load_rollouts(tmp_path, RUN_ID, require_grades=False,
                                    prefer_parquet=False)

    assert via_parquet.source == "parquet"
    assert via_jsonl.source == "jsonl"
    assert list(via_parquet.rows) == list(via_jsonl.rows)


# -- the four refusals -------------------------------------------------------


def test_refuses_an_unsealed_run(tmp_path: Path):
    write_run(tmp_path, [graded(make_generation(0))], sealed=False)
    with pytest.raises(StoreReadError, match="_MANIFEST"):
        store.load_rollouts(tmp_path, RUN_ID)


def test_refuses_a_run_that_does_not_exist(tmp_path: Path):
    with pytest.raises(StoreReadError, match="latest"):
        store.load_rollouts(tmp_path, "2026-01-01-0000000-000000")


@pytest.mark.parametrize("splits", [("test",), ("train", "test"), ("val", "test")])
def test_the_test_split_is_not_loadable(graded_run: Path, splits):
    with pytest.raises(SplitError, match="opened exactly once"):
        store.load_rollouts(graded_run, RUN_ID, splits=splits)


def test_there_is_no_flag_that_opens_the_test_split():
    """The absence is the enforcement, so assert the absence."""
    import inspect

    parameters = inspect.signature(store.load_rollouts).parameters
    assert "allow_test" not in parameters
    assert "test" not in store.ALLOWED_SPLITS


def test_refuses_an_ungraded_run_by_default(tmp_path: Path):
    """Every real row in the repo today is in exactly this state."""
    write_run(tmp_path, [make_generation(i).to_row() for i in range(4)])
    with pytest.raises(UngradedRunError, match="no hidden-test outcome"):
        store.load_rollouts(tmp_path, RUN_ID)


def test_an_ungraded_run_can_still_be_inspected(tmp_path: Path):
    write_run(tmp_path, [make_generation(i).to_row() for i in range(4)])
    data = store.load_rollouts(tmp_path, RUN_ID, require_grades=False)
    assert len(data) == 4
    assert data.labels == {}


def test_a_mixed_version_store_is_refused_rather_than_averaged(tmp_path: Path):
    rows = [graded(make_generation(0)), graded(make_generation(1))]
    rows[1]["schema_version"] = 2
    write_run(tmp_path, rows)
    with pytest.raises(SchemaVersionError):
        store.load_rollouts(tmp_path, RUN_ID)


# -- the structural leak guard -----------------------------------------------


def test_hidden_columns_are_absent_from_every_row(graded_run: Path):
    data = store.load_rollouts(graded_run, RUN_ID)
    for row in data.rows:
        assert "hidden_passed" not in row
        assert "hidden_total" not in row


def test_labels_come_back_separately_and_carry_the_outcome(graded_run: Path):
    data = store.load_rollouts(graded_run, RUN_ID)
    assert len(data.labels) == len(data)
    for row in data.rows:
        label = data.label_for(row["rollout_id"])
        assert label.task_id == row["task_id"]
        assert label.solved is (label.hidden_passed == label.hidden_total)


def test_the_visible_outcome_survives_because_it_is_not_a_label(graded_run: Path):
    """D1's whole premise: the visible tests are observable at decision time."""
    data = store.load_rollouts(graded_run, RUN_ID)
    assert all(row["visible_total"] == 3 for row in data.rows)


# -- cost -------------------------------------------------------------------


def test_no_costing_is_attached_unless_one_is_pinned(graded_run: Path):
    data = store.load_rollouts(graded_run, RUN_ID)
    assert data.has_cost is False
    assert all(row["gpu_seconds"] is None for row in data.rows)


def test_a_pinned_costing_joins_by_rollout_id(tmp_path: Path):
    rows = [graded(make_generation(i)) for i in range(4)]
    write_run(tmp_path, rows)
    write_cost(tmp_path, rows, "abc12345")

    data = store.load_rollouts(tmp_path, RUN_ID, cost_fingerprint="abc12345")
    assert data.has_cost is True
    assert all(row["gpu_seconds"] == 0.5 for row in data.rows)
    assert all(row["imputed_latency_s"] == 1.25 for row in data.rows)


def test_a_costing_that_does_not_exist_lists_the_ones_that_do(tmp_path: Path):
    rows = [graded(make_generation(0))]
    write_run(tmp_path, rows)
    write_cost(tmp_path, rows, "abc12345")
    with pytest.raises(StoreReadError, match="abc12345"):
        store.load_rollouts(tmp_path, RUN_ID, cost_fingerprint="deadbeef")


def test_costings_are_listed_not_resolved(tmp_path: Path):
    """Two costings of one run coexist, so 'the cost' is not a request."""
    rows = [graded(make_generation(0))]
    write_run(tmp_path, rows)
    write_cost(tmp_path, rows, "a100feed")
    write_cost(tmp_path, rows, "h100beef")
    assert store.list_cost_fingerprints(tmp_path, RUN_ID) == ["a100feed", "h100beef"]


# -- the task join -----------------------------------------------------------


def write_tasks(tmp_path: Path, n: int = 4, *, with_hidden: bool = True) -> Path:
    path = tmp_path / "tasks.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            record: dict[str, Any] = {
                "task_id": f"mbpp/{i}",
                "prompt": f"Write a function that returns {i}.",
                "entrypoint": f"f{i}",
                "visible_tests": f"assert f{i}() == {i}",
            }
            if with_hidden:
                # The full suite, which is what R2 grades against and what must
                # never cross the join.
                record["tests"] = f"assert f{i}() == {i}\nassert f{i}() != -1"
                record["metadata"] = {"canonical_solution": f"def f{i}(): return {i}"}
            fh.write(json.dumps(record) + "\n")
    return path


def test_the_task_join_supplies_the_prompt_the_row_does_not_carry(tmp_path: Path):
    rows = [graded(make_generation(i)) for i in range(4)]
    write_run(tmp_path, rows)
    tasks = write_tasks(tmp_path)

    data = store.load_rollouts(tmp_path, RUN_ID, tasks_path=tasks)
    assert data.rows[0]["task_prompt"].startswith("Write a function")
    assert data.rows[0]["task_entrypoint"] == "f0"
    assert data.rows[0]["task_visible_tests"] == "assert f0() == 0"


def test_the_task_join_leaves_the_hidden_suite_and_the_solution_behind(tmp_path: Path):
    """The join adds exactly three fields. The full suite is not one of them."""
    rows = [graded(make_generation(i)) for i in range(4)]
    write_run(tmp_path, rows)
    tasks = write_tasks(tmp_path)

    without = store.load_rollouts(tmp_path, RUN_ID)
    joined = store.load_rollouts(tmp_path, RUN_ID, tasks_path=tasks)

    added = set(joined.rows[0]) - set(without.rows[0])
    assert added == set(store.TASK_FIELDS)
    for row in joined.rows:
        assert set(row) & {"tests", "canonical_solution", "metadata"} == set()


def test_a_corpus_that_does_not_match_the_run_is_refused(tmp_path: Path):
    rows = [graded(make_generation(i)) for i in range(4)]
    write_run(tmp_path, rows)
    tasks = write_tasks(tmp_path, n=2)
    with pytest.raises(StoreReadError, match="does not match"):
        store.load_rollouts(tmp_path, RUN_ID, tasks_path=tasks)


# -- splits, damage, and the small helpers -----------------------------------


def test_only_the_requested_splits_come_back(tmp_path: Path):
    rows = [graded(make_generation(0, split="train")),
            graded(make_generation(1, split="val")),
            graded(make_generation(2, split="test"))]
    write_run(tmp_path, rows)

    data = store.load_rollouts(tmp_path, RUN_ID, splits=("val",))
    assert data.counts_by_split() == {"val": 1}
    assert "test" not in data.counts_by_split()


def test_test_split_rows_are_dropped_even_when_present_in_the_run(tmp_path: Path):
    """R1 generates on the test split today. The rows exist; R3 never sees them."""
    rows = [graded(make_generation(0, split="train")),
            graded(make_generation(1, split="test")),
            graded(make_generation(2, split="test"))]
    write_run(tmp_path, rows)

    data = store.load_rollouts(tmp_path, RUN_ID)
    assert len(data) == 1
    assert data.labels.keys() == {data.rows[0]["rollout_id"]}


def test_a_torn_final_line_is_skipped(tmp_path: Path):
    rows = [graded(make_generation(i)) for i in range(3)]
    write_run(tmp_path, rows)
    part = next((tmp_path / RUN_ID / "generations").glob("part-*.jsonl"))
    with part.open("a", encoding="utf-8") as fh:
        fh.write('{"task_id": "mbpp/9", "arm": "dir')

    data = store.load_rollouts(tmp_path, RUN_ID)
    assert len(data) == 3


def test_a_damaged_middle_line_is_refused(tmp_path: Path):
    """Interrupted is recoverable. Damaged is not, and must not read as a run."""
    rows = [graded(make_generation(i)) for i in range(3)]
    write_run(tmp_path, rows)
    part = next((tmp_path / RUN_ID / "generations").glob("part-*.jsonl"))
    lines = part.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"task_id": "mbpp/1", "arm": '
    part.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(StoreReadError, match="damaged"):
        store.load_rollouts(tmp_path, RUN_ID)


def test_publishable_follows_the_manifest(tmp_path: Path):
    rows = [graded(make_generation(0))]
    write_run(tmp_path, rows, publishable=False)
    assert store.load_rollouts(tmp_path, RUN_ID).publishable is False


def test_counts_describe_the_selection_not_the_run(graded_run: Path):
    data = store.load_rollouts(graded_run, RUN_ID)
    assert data.counts_by_arm() == {"direct_large": 8, "direct_small": 8}
    assert sum(data.counts_by_split().values()) == len(data)
