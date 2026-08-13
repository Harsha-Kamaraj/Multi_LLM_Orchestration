"""`orch-workers` — one command, and it has to work.

A sweep that needs a checklist gets run inconsistently, so the CLI is the
surface everyone else uses and every subcommand is smoke-tested here.
"""

from __future__ import annotations

import pytest

from orchestrator.workers.cli import main


def run(*argv) -> int:
    return main([str(a) for a in argv])


@pytest.fixture
def swept(corpus_path, runs_root, capsys):
    """A completed sweep, with its run_id."""
    assert run("--runs-root", runs_root, "sweep", "--tasks", corpus_path,
               "--backend", "mock", "--seeds", 0, 1, "--batch-size", 8,
               "--coefficients", corpus_path.parent / "absent.json") == 0
    out = capsys.readouterr().out
    run_id = [ln.split(": ", 1)[1] for ln in out.splitlines()
              if ln.startswith("run_id: ")][0]
    return run_id


def test_sweep_reports_its_run_id(swept):
    from orchestrator.workers.runid import is_valid

    assert is_valid(swept)


def test_runs_lists_the_sealed_run(runs_root, swept, capsys):
    assert run("--runs-root", runs_root, "runs") == 0
    assert swept in capsys.readouterr().out


def test_show_prints_a_manifest_summary(runs_root, swept, capsys):
    assert run("--runs-root", runs_root, "show", "--run", swept) == 0
    out = capsys.readouterr().out
    assert "truncation" in out and "by arm" in out and "direct_small" in out


def test_show_json_emits_the_raw_manifest(runs_root, swept, capsys):
    import json

    assert run("--runs-root", runs_root, "show", "--run", swept, "--json") == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["run_id"] == swept and manifest["n_rows"] == 96


def test_characterize_writes_coefficients(tmp_path, capsys):
    out = tmp_path / "cost_coefficients.json"
    code = run("characterize", "--backend", "mock",
               "--backend-option", "mode=serving", "--out", out,
               "--concurrency", 1, "--repeats", 3, "--hardware", "mock-host",
               "--usd-per-gpu-hour", 1.10, "--allow-approx-tokens",
               "--threshold", 0.5)
    assert code == 0 and out.exists()
    assert "ms/tok" in capsys.readouterr().out


def test_characterize_exits_nonzero_when_imputation_is_untrustworthy(tmp_path):
    """A failing R-squared must not be scripted past."""
    assert run("characterize", "--backend", "mock",
               "--backend-option", "mode=serving",
               "--out", tmp_path / "c.json", "--concurrency", 1,
               "--allow-approx-tokens", "--threshold", 0.9999) == 1


def test_impute_writes_a_sidecar(tmp_path, runs_root, swept, capsys):
    coefficients = tmp_path / "c.json"
    assert run("characterize", "--backend", "mock",
               "--backend-option", "mode=serving", "--out", coefficients,
               "--concurrency", 1, "--allow-approx-tokens",
               "--threshold", 0.1) == 0
    assert run("--runs-root", runs_root, "impute", "--run", swept,
               "--coefficients", coefficients) == 0
    assert "GPU-seconds" in capsys.readouterr().out


def test_validate_refuses_to_use_sweep_rows(tmp_path, runs_root, swept, capsys):
    """A batched sweep's wall-clock measures queue depth and cannot validate
    a latency model. This is the safety property of the whole design."""
    coefficients = tmp_path / "c.json"
    run("characterize", "--backend", "mock", "--backend-option", "mode=serving",
        "--out", coefficients, "--concurrency", 1, "--allow-approx-tokens",
        "--threshold", 0.1)
    assert run("--runs-root", runs_root, "validate", "--run", swept,
               "--coefficients", coefficients) == 1
    assert "queue depth" in capsys.readouterr().err


def test_extract_debugs_a_response(tmp_path, capsys):
    path = tmp_path / "response.txt"
    path.write_text("```python\nimport math\n```\n```python\ndef add(a, b):\n"
                    "    return math.floor(a + b)\n```")
    assert run("extract", path, "--entrypoint", "add") == 0
    out = capsys.readouterr().out
    assert "fenced_imports_merged" in out and "import math" in out


def test_extract_exits_nonzero_on_unparseable_output(tmp_path):
    path = tmp_path / "response.txt"
    path.write_text("I can't help with that.")
    assert run("extract", path) == 1


def test_a_missing_task_manifest_is_an_actionable_error(tmp_path, capsys):
    """Expected failures print their message; a traceback would bury it."""
    assert run("sweep", "--tasks", tmp_path / "absent.jsonl", "--backend", "mock") == 2
    assert "R2 owns" in capsys.readouterr().err


def test_missing_coefficients_are_an_actionable_error(runs_root, swept, tmp_path, capsys):
    assert run("--runs-root", runs_root, "validate", "--run", swept,
               "--coefficients", tmp_path / "absent.json") == 2
    assert "characterization pass" in capsys.readouterr().err


def test_backend_options_are_type_coerced():
    """`gpu_memory_utilization=0.9` must arrive as a float, not a string."""
    from orchestrator.workers.cli import _backend_options

    parsed = _backend_options(["gpu_memory_utilization=0.9", "enforce_eager=true",
                               "base_url=http://host:8000/v1"])
    assert parsed == {
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "base_url": "http://host:8000/v1",
    }


def test_a_malformed_backend_option_is_rejected():
    from orchestrator.workers.cli import _backend_options

    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _backend_options(["novalue"])


def test_an_unknown_arm_is_rejected_by_the_parser(corpus_path):
    with pytest.raises(SystemExit):
        run("sweep", "--tasks", corpus_path, "--arms", "direct_medium")


def test_runs_reports_an_empty_store(tmp_path, capsys):
    assert run("--runs-root", tmp_path / "runs", "runs") == 0
    assert "no runs" in capsys.readouterr().out


def test_validate_prints_its_report_on_eligible_rows(tmp_path, corpus_path,
                                                     runs_root, capsys):
    """The success path, which the refusal tests never reach: they return
    before printing. `validate` is the definition-of-done command, and it
    printed a report that could not be produced -- `summary` is a property,
    and calling it raised TypeError against every real run."""
    from orchestrator.workers.characterize import build_probes, run_and_save
    from orchestrator.workers.backends import get_backend

    coefficients = tmp_path / "cost_coefficients.json"
    run_and_save(get_backend("mock", mode="serving"), coefficients,
                 concurrencies=(1,), probes=build_probes(repeats=6),
                 require_exact_tokens=False)

    # A serving-mode sweep at the reference batch is what validate accepts.
    assert run("--runs-root", runs_root, "sweep", "--tasks", corpus_path,
               "--backend", "mock", "--backend-option", "mode=serving",
               "--seeds", 0, "--batch-size", 1,
               "--coefficients", coefficients) == 0
    run_id = [ln.split(": ", 1)[1] for ln in capsys.readouterr().out.splitlines()
              if ln.startswith("run_id: ")][0]

    rc = run("--runs-root", runs_root, "validate", "--run", run_id,
             "--coefficients", coefficients)
    out = capsys.readouterr().out
    assert "R^2=" in out and ("[PASS]" in out or "[FAIL]" in out)
    assert rc in (0, 1)
