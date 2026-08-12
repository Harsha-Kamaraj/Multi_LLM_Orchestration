from __future__ import annotations

import json

import pytest

from orchestrator.graders.cli import main
from orchestrator.graders.rollout_store import read_manifest, read_rows
from orchestrator.workers.sweep import SweepConfig, run_sweep

from .conftest import HIDDEN_TESTS, VISIBLE_TESTS


@pytest.fixture
def tasks_file(tmp_path):
    path = tmp_path / "tasks.jsonl"
    record = {
        "task_id": "triage-1",
        "prompt": "sign classifier",
        "entrypoint": "triage",
        "tests": HIDDEN_TESTS,
        "visible_tests": VISIBLE_TESTS,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def _sealed_generation_run(tmp_path, tasks_file) -> tuple[str, object]:
    runs_root = tmp_path / "runs"
    report = run_sweep(SweepConfig(
        tasks_path=tasks_file, out_root=runs_root,
        arms=("direct_small",), seeds=(0,), backend="mock",
    ))
    return report.run_id, runs_root


def test_grade_one_via_cli(tasks_file, tmp_path, capsys):
    # Hardcoded to exactly the two cases the weak visible tests check;
    # wrong on the hidden-only cases (0, 100, -100, 1, -1).
    code_file = tmp_path / "sol.py"
    code_file.write_text(
        "def triage(n):\n"
        "    if n == 5:\n        return 2\n"
        "    if n == -5:\n        return 0\n"
        "    return 2\n",
        encoding="utf-8",
    )

    rc = main([
        "grade", "one",
        "--tasks", str(tasks_file), "--task-id", "triage-1",
        "--code-file", str(code_file), "--backend", "subprocess",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["visible_passed"] == 1  # matches the weak visible test exactly
    assert out["hidden_passed"] == 0   # fails triage(0)/(100)/(-100)/(1)/(-1)


def test_grade_run_refuses_bare_subprocess(tasks_file, tmp_path):
    run_id, runs_root = _sealed_generation_run(tmp_path, tasks_file)
    with pytest.raises(SystemExit):
        main([
            "grade", "run", "--run-id", run_id, "--tasks", str(tasks_file),
            "--runs-root", str(runs_root), "--backend", "subprocess",
        ])
    # no rollouts store should have been created by the refused attempt
    assert not (runs_root / run_id / "rollouts").exists()


def test_grade_run_end_to_end(tasks_file, tmp_path, capsys):
    run_id, runs_root = _sealed_generation_run(tmp_path, tasks_file)

    rc = main([
        "grade", "run", "--run-id", run_id, "--tasks", str(tasks_file),
        "--runs-root", str(runs_root), "--backend", "subprocess",
        "--unsafe-subprocess-local-only",
    ])
    assert rc == 0

    manifest = read_manifest(runs_root, run_id)
    assert manifest["n_rows"] == 1
    rows = list(read_rows(runs_root, run_id))
    assert rows[0]["task_id"] == "triage-1"
    assert rows[0]["error_class"] in {"none", "runtime_error", "syntax_error", "empty_code"}
    # never mutated R1's own generations manifest
    from orchestrator.workers.store import read_manifest as read_gen_manifest
    gen_manifest = read_gen_manifest(runs_root, run_id)
    assert gen_manifest["n_rows"] == 1
