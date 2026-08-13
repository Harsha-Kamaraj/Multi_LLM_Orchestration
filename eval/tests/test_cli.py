"""The CLI, including the guards that must survive being invoked by a human.

The test-split lock is the one that matters here: a command-line flag is the
easiest place for a discipline to erode, because it takes one typed argument to
undo it. So the lock is tested at the CLI boundary, not only in the library.
"""

from __future__ import annotations

import json

import pytest

from eval.cli import main
from schemas import SynthConfig, generate


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A sealed run on disk, in the layout R1's store writes."""
    root = tmp_path_factory.mktemp("runs")
    synth = generate(SynthConfig(n_tasks=120, seeds=3), seed=5)
    run_id = synth.run_id

    generations = root / run_id / "generations"
    generations.mkdir(parents=True)
    with open(generations / "part-000.jsonl", "w", encoding="utf-8") as fh:
        for row in synth.rows:
            fh.write(json.dumps(row) + "\n")
    (root / run_id / "_MANIFEST.json").write_text('{"rows": %d}' % len(synth.rows))
    return root, run_id


def test_report_runs_and_prints_a_table(run_dir, capsys):
    root, run_id = run_dir
    code = main(["report", "--root", str(root), "--run-id", run_id,
                 "--resamples", "200"])
    out = capsys.readouterr().out
    assert code == 0
    assert "verifier_gated_cascade" in out
    assert "reference" in out


def test_report_seals_output_when_asked(run_dir, tmp_path):
    root, run_id = run_dir
    out = tmp_path / "sealed"
    assert main(["report", "--root", str(root), "--run-id", run_id,
                 "--resamples", "100", "--out", str(out)]) == 0
    assert (out / "results.json").exists()
    assert (out / "_MANIFEST.json").exists()


def test_unsealed_run_is_refused(tmp_path, capsys):
    """A run killed midway looks complete, because the rows that exist are."""
    root = tmp_path / "runs"
    synth = generate(SynthConfig(n_tasks=30, seeds=3), seed=2)
    gen = root / synth.run_id / "generations"
    gen.mkdir(parents=True)
    with open(gen / "part-000.jsonl", "w", encoding="utf-8") as fh:
        for row in synth.rows:
            fh.write(json.dumps(row) + "\n")
    # No _MANIFEST.json written.

    code = main(["report", "--root", str(root), "--run-id", synth.run_id])
    assert code == 2
    assert "_MANIFEST" in capsys.readouterr().err


def test_missing_run_is_an_error(tmp_path, capsys):
    code = main(["report", "--root", str(tmp_path), "--run-id",
                 "2026-01-01-0f1ced0-abcdef"])
    assert code == 2
    assert "no run directory" in capsys.readouterr().err


def test_test_split_requires_a_prereg_flag(run_dir):
    """One typed argument is all it takes to erode a discipline, so the lock is
    tested where a human actually types."""
    root, run_id = run_dir
    with pytest.raises(SystemExit, match="unlock-prereg"):
        main(["report", "--root", str(root), "--run-id", run_id,
              "--splits", "train,val,test"])


def test_test_split_requires_the_prereg_to_exist(run_dir, tmp_path, capsys):
    root, run_id = run_dir
    code = main(["report", "--root", str(root), "--run-id", run_id,
                 "--splits", "test", "--unlock-prereg", str(tmp_path / "no.md")])
    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_test_split_opens_with_a_real_prereg(run_dir, tmp_path, capsys):
    root, run_id = run_dir
    prereg = tmp_path / "prereg.md"
    prereg.write_text("metric: solved\ntest: mcnemar\ncorrection: BH\n")
    code = main(["report", "--root", str(root), "--run-id", run_id,
                 "--splits", "test", "--unlock-prereg", str(prereg),
                 "--resamples", "100"])
    assert code == 0
    assert "always_small" in capsys.readouterr().out


def test_audit_exits_zero_on_a_clean_store(run_dir, capsys):
    root, run_id = run_dir
    assert main(["audit", "--root", str(root), "--run-id", run_id]) == 0
    assert "split_disjointness" in capsys.readouterr().out


def test_power_reports_a_task_count(capsys):
    assert main(["power", "--discordance", "0.18", "--odds-ratio", "1.45"]) == 0
    out = capsys.readouterr().out
    assert "tasks needed:" in out
    assert "DISCORDANT" in out, "the caveat must travel with the number"


def test_golden_passes_against_a_fresh_report(run_dir, tmp_path, capsys):
    root, run_id = run_dir
    out = tmp_path / "sealed"
    main(["report", "--root", str(root), "--run-id", run_id,
          "--resamples", "100", "--seed", "3", "--out", str(out)])

    code = main(["golden", "--root", str(root), "--run-id", run_id,
                 "--resamples", "100", "--seed", "3",
                 "--golden", str(out / "results.json")])
    assert code == 0
    assert "reproduces exactly" in capsys.readouterr().out


def test_golden_fails_when_a_number_moves(run_dir, tmp_path, capsys):
    root, run_id = run_dir
    out = tmp_path / "sealed"
    main(["report", "--root", str(root), "--run-id", run_id,
          "--resamples", "100", "--seed", "3", "--out", str(out)])

    payload = json.loads((out / "results.json").read_text())
    payload["policies"]["always_large"]["accuracy"] += 0.02
    (out / "results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n"
    )

    code = main(["golden", "--root", str(root), "--run-id", run_id,
                 "--resamples", "100", "--seed", "3",
                 "--golden", str(out / "results.json")])
    assert code == 1
    err = capsys.readouterr().err
    assert "BREAKING-GOLDEN" in err
    assert "always_large" in err


def test_run_id_is_required():
    with pytest.raises(SystemExit):
        main(["report", "--root", "runs"])
