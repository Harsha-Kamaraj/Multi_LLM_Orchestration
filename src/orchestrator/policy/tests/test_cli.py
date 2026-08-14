"""`orch-policy`, and the exit codes that make the gate a tripwire.

The exit-code contract is the part worth testing hardest. A gate that prints a
failure and exits zero is not a gate — it is a log line someone has to remember
to read. And INCONCLUSIVE exiting zero would be worse than either: it would
turn "we could not tell" into "it passed", which is the specific way a
threshold quietly stops meaning anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.policy import fixtures
from orchestrator.policy.cli import (
    EXIT_ERROR, EXIT_GATE_FAILED, EXIT_INCONCLUSIVE, EXIT_OK, main,
)
from schemas.synth import SynthConfig


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> fixtures.Fixture:
    root = tmp_path_factory.mktemp("cli")
    return fixtures.write_fixture(root, SynthConfig(n_tasks=400, seeds=3))


def gate_argv(fx: fixtures.Fixture, *extra: str) -> list[str]:
    return [
        "gate", "--root", str(fx.root), "--run", fx.run_id,
        "--tasks", str(fx.tasks_path), "--cost", str(fx.cost_fingerprint),
        "--resamples", "150",
        *extra,
    ]


# -- the happy path ----------------------------------------------------------


def test_gate_reports_both_decision_points(fx, capsys):
    assert main(gate_argv(fx)) == EXIT_OK
    out = capsys.readouterr().out
    assert "AUC_D0" in out and "AUC_D1" in out
    assert "D1 - D0" in out


def test_gate_can_measure_one_decision_point(fx, capsys):
    assert main(gate_argv(fx, "--decision-point", "D1")) == EXIT_OK
    out = capsys.readouterr().out
    assert "AUC_D1" in out
    assert "AUC_D0" not in out


def test_every_reported_number_carries_an_interval(fx, capsys):
    """No bare means. A merge blocker, and it applies to R3's own output."""
    main(gate_argv(fx))
    for line in capsys.readouterr().out.splitlines():
        if "AUC_D" in line and "=" in line:
            assert "[" in line and "," in line and "]" in line, line


def test_the_probe_is_opt_in(fx, capsys):
    """Its features oblige the policy to pay for the draws they read."""
    main(gate_argv(fx, "--decision-point", "D1"))
    plain = capsys.readouterr().out
    main(gate_argv(fx, "--decision-point", "D1", "--with-probe"))
    probed = capsys.readouterr().out

    def n_features(text: str) -> int:
        return int(text.split(" features")[0].split()[-1])

    assert n_features(probed) > n_features(plain)


# -- exit codes --------------------------------------------------------------


def test_a_passing_gate_exits_zero(fx):
    assert main(gate_argv(fx)) == EXIT_OK


def test_a_failed_hard_stop_exits_non_zero(fx, monkeypatch, capsys):
    """`AUC_D1 < 0.75` ends the project, so it must not exit zero."""
    from orchestrator.policy import cli, gate
    from eval.stats import Interval

    real = gate.measure_gate

    def failing(data, point, **kwargs):
        result = real(data, point, **kwargs)
        if point != "D1":
            return result
        return type(result)(**{**result.__dict__,
                               "auc": Interval(0.61, 0.55, 0.66)})

    monkeypatch.setattr(cli.gate, "measure_gate", failing)
    assert main(gate_argv(fx)) == EXIT_GATE_FAILED
    assert "premise is" in capsys.readouterr().out


def test_an_inconclusive_gate_does_not_exit_zero(fx, monkeypatch):
    """"Could not tell" must never be reported to a caller as "passed"."""
    from orchestrator.policy import cli, gate
    from eval.stats import Interval

    real = gate.measure_gate

    def straddling(data, point, **kwargs):
        result = real(data, point, **kwargs)
        if point != "D1":
            return result
        return type(result)(**{**result.__dict__,
                               "auc": Interval(0.76, 0.71, 0.81)})

    monkeypatch.setattr(cli.gate, "measure_gate", straddling)
    assert main(gate_argv(fx)) == EXIT_INCONCLUSIVE


def test_a_weak_d0_alone_is_not_fatal(fx, monkeypatch):
    """D0 failing means pre-generation routing is dead — expected, not fatal."""
    from orchestrator.policy import cli, gate
    from eval.stats import Interval

    real = gate.measure_gate

    def weak_d0(data, point, **kwargs):
        result = real(data, point, **kwargs)
        if point != "D0":
            return result
        return type(result)(**{**result.__dict__,
                               "auc": Interval(0.52, 0.48, 0.56)})

    monkeypatch.setattr(cli.gate, "measure_gate", weak_d0)
    assert main(gate_argv(fx)) == EXIT_OK


# -- refusals reach the caller as messages, not tracebacks -------------------


def test_an_unknown_run_is_an_error_not_a_traceback(fx, capsys):
    code = main(["gate", "--root", str(fx.root),
                 "--run", "2020-01-01-0000000-000000"])
    assert code == EXIT_ERROR
    assert "latest" in capsys.readouterr().err


def test_an_unsealed_run_is_refused_by_the_command(tmp_path, capsys):
    fx = fixtures.write_fixture(tmp_path, SynthConfig(n_tasks=40, seeds=2),
                                sealed=False)
    code = main(["gate", "--root", str(fx.root), "--run", fx.run_id])
    assert code == EXIT_ERROR
    assert "_MANIFEST" in capsys.readouterr().err


def test_building_d0_without_the_task_manifest_says_why(fx, capsys):
    """The rollout row carries no prompt, and the error has to say so."""
    code = main(["gate", "--root", str(fx.root), "--run", fx.run_id,
                 "--decision-point", "D0", "--resamples", "50"])
    assert code == EXIT_ERROR
    assert "carries no prompt" in capsys.readouterr().err


def test_there_is_no_flag_that_opens_the_test_split():
    """The absence is the enforcement, so assert the absence."""
    from orchestrator.policy.cli import build_parser

    text = build_parser().format_help()
    for option in ("--split", "--test", "--allow-test"):
        assert option not in text


# -- the written verdict -----------------------------------------------------


def test_the_verdict_can_be_written_as_json(fx, tmp_path, capsys):
    out = tmp_path / "gate.json"
    assert main(gate_argv(fx, "--out", str(out))) == EXIT_OK

    payload = json.loads(out.read_text())
    for point in ("D0", "D1"):
        assert payload[point]["run_id"] == fx.run_id
        assert payload[point]["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")
        assert set(payload[point]["auc"]) >= {"point", "low", "high"}
    assert payload["cost_fingerprint"] == fx.cost_fingerprint
    assert payload["publishable"] is True


def test_the_verdict_records_what_it_was_measured_from(fx, tmp_path):
    """Pinned to a run and a costing, so the number can be reproduced."""
    out = tmp_path / "gate.json"
    main(gate_argv(fx, "--out", str(out)))
    payload = json.loads(out.read_text())
    assert payload["tasks_path"] == str(fx.tasks_path)
    assert payload["D1"]["run_id"] == fx.run_id


# -- the other subcommands ---------------------------------------------------


def test_fixture_writes_a_measurable_run(tmp_path, capsys):
    assert main(["fixture", "--out", str(tmp_path), "--tasks", "60",
                 "--seeds", "2"]) == EXIT_OK
    out = capsys.readouterr().out
    run_id = out.split("run_id")[1].split()[0]
    assert (tmp_path / run_id / "_MANIFEST.json").exists()
    assert (tmp_path / run_id / "tasks.jsonl").exists()


def test_runs_lists_only_sealed_runs(fx, capsys):
    assert main(["runs", "--root", str(fx.root)]) == EXIT_OK
    out = capsys.readouterr().out
    assert fx.run_id in out
    assert str(fx.cost_fingerprint) in out


def test_runs_on_an_empty_store_says_so(tmp_path, capsys):
    assert main(["runs", "--root", str(tmp_path / "nope")]) == EXIT_OK
    assert "no store" in capsys.readouterr().out


def test_runs_marks_a_dirty_run_unpublishable_despite_its_manifest(tmp_path, capsys):
    """The listing is where an operator looks before reporting from a run.

    A `-dirty` sweep keeps `publishable: true` in its manifest — the suffix is
    what overrides it, and `store.is_publishable` is the one place that rule
    lives. Reading the manifest key directly here would have shown the run as
    publishable in the only view an operator actually consults.

    Builds its own run rather than using the module-scoped one: this test
    renames the directory, and a module fixture is shared with every test that
    comes after it.
    """
    fx = fixtures.write_fixture(tmp_path, SynthConfig(n_tasks=40, seeds=2))
    dirty = f"{fx.run_id}-dirty"
    (fx.root / fx.run_id).rename(fx.root / dirty)

    manifest = json.loads((fx.root / dirty / "_MANIFEST.json").read_text())
    assert manifest["publishable"] is True, "the suffix must be doing the work"

    assert main(["runs", "--root", str(fx.root)]) == EXIT_OK
    line = next(ln for ln in capsys.readouterr().out.splitlines()
                if ln.startswith(dirty))
    assert "[not publishable]" in line


# -- the graded layer --------------------------------------------------------


def test_the_gate_reads_the_graded_layer_of_a_split_run(tmp_path, capsys):
    """The labels live in R2's `rollouts/`, and nothing has to say so."""
    fx = fixtures.write_fixture(tmp_path, SynthConfig(n_tasks=200, seeds=3),
                                layout="split")
    code = main(["gate", "--root", str(fx.root), "--run", fx.run_id,
                 "--tasks", str(fx.tasks_path), "--decision-point", "D1",
                 "--resamples", "80"])
    assert code in (EXIT_OK, EXIT_INCONCLUSIVE)
    assert "AUC_D1" in capsys.readouterr().out


def test_asking_for_the_ungraded_layer_reports_it_as_unusable(tmp_path, capsys):
    fx = fixtures.write_fixture(tmp_path, SynthConfig(n_tasks=60, seeds=2),
                                layout="split")
    code = main(["gate", "--root", str(fx.root), "--run", fx.run_id,
                 "--layer", "generations", "--resamples", "20"])
    assert code == EXIT_ERROR
    assert "no hidden-test outcome" in capsys.readouterr().err


def test_runs_distinguishes_graded_from_ungraded(tmp_path, capsys):
    """Whether R2 has been through is the first thing to know about a run."""
    fixtures.write_fixture(tmp_path, SynthConfig(n_tasks=40, seeds=2),
                           layout="split")
    main(["runs", "--root", str(tmp_path)])
    assert "graded" in capsys.readouterr().out

    other = tmp_path / "plain"
    fixtures.write_fixture(other, SynthConfig(n_tasks=40, seeds=2))
    main(["runs", "--root", str(other)])
    assert "UNGRADED" in capsys.readouterr().out


def test_fixture_can_write_the_two_layer_layout(tmp_path, capsys):
    assert main(["fixture", "--out", str(tmp_path), "--tasks", "40",
                 "--seeds", "2", "--layout", "split"]) == EXIT_OK
    out = capsys.readouterr().out
    run_id = out.split("run_id")[1].split()[0]
    assert (tmp_path / run_id / "_ROLLOUT_MANIFEST.json").exists()
    assert (tmp_path / run_id / "rollouts").is_dir()


# -- train -------------------------------------------------------------------


def train_argv(fx: fixtures.Fixture, *extra: str) -> list[str]:
    return [
        "train", "--root", str(fx.root), "--run", fx.run_id,
        "--tasks", str(fx.tasks_path), "--cost", str(fx.cost_fingerprint),
        *extra,
    ]


def test_train_reports_every_arm(fx, capsys):
    main(train_argv(fx))
    out = capsys.readouterr().out
    assert "arms: large, small" in out
    for arm in ("large", "small"):
        assert f"arm '{arm}'" in out


def test_train_names_which_calibration_map_shipped(fx, capsys):
    """`applied=False` is a decision, not an absence, and has to be visible."""
    main(train_argv(fx))
    out = capsys.readouterr().out
    assert "using isotonic" in out or "using identity" in out


def test_train_writes_the_artifact_set(fx, tmp_path):
    from orchestrator.policy import heads

    out = tmp_path / "policy"
    assert main(train_argv(fx, "--out", str(out))) in (EXIT_OK, EXIT_GATE_FAILED)
    for name in (heads.POLICY_PICKLE, heads.HEADS_MANIFEST,
                 heads.CALIBRATION_FILE, heads.FEATURE_SPEC_FILE):
        assert (out / name).exists(), name


def test_a_miscalibrated_policy_does_not_exit_zero(fx, monkeypatch, capsys):
    """Same contract as the gate: shipping an uncalibrated `P_pass` quietly is
    worse than failing loudly, because λ silently stops meaning anything."""
    from orchestrator.policy import calibration

    monkeypatch.setattr(calibration, "ECE_TARGET", 0.0)
    assert main(train_argv(fx)) == EXIT_GATE_FAILED


def test_train_can_target_d0(fx, capsys):
    main(train_argv(fx, "--decision-point", "D0"))
    assert "policy for D0" in capsys.readouterr().out


def test_train_refuses_an_unknown_run(fx, capsys):
    argv = ["train", "--root", str(fx.root), "--run", "2026-01-01-abcdefg-000000"]
    assert main(argv) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_there_is_no_train_flag_that_opens_the_test_split():
    """The absence is the enforcement, as everywhere else in this package."""
    from orchestrator.policy.cli import build_parser

    text = build_parser().format_help()
    train = [a for a in build_parser()._subparsers._group_actions[0].choices]
    assert "train" in train
    assert "--test" not in text and "test-split" not in text


# -- decide ------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(fx, tmp_path_factory) -> Path:
    """A D0 policy on disk, since `decide` takes a policy rather than fitting one."""
    out = tmp_path_factory.mktemp("trained") / "policy"
    main(["train", "--root", str(fx.root), "--run", fx.run_id,
          "--tasks", str(fx.tasks_path), "--decision-point", "D0",
          "--out", str(out)])
    return out


def decide_argv(fx: fixtures.Fixture, policy: Path, *extra: str) -> list[str]:
    return [
        "decide", "--root", str(fx.root), "--run", fx.run_id,
        "--tasks", str(fx.tasks_path), "--policy", str(policy),
        "--coefficients", str(fx.coefficients_path),
        *extra,
    ]


def test_decide_sweeps_the_whole_frozen_grid(fx, trained, capsys):
    from orchestrator.policy.decide import LAMBDA_GRID

    assert main(decide_argv(fx, trained)) == EXIT_OK
    out = capsys.readouterr().out
    assert out.count("lambda=") == len(LAMBDA_GRID)


def test_decide_writes_the_artifacts_r4_reads(fx, trained, tmp_path):
    from orchestrator.policy import decide

    out = tmp_path / "decisions"
    assert main(decide_argv(fx, trained, "--out", str(out))) == EXIT_OK
    assert (out / decide.DECISIONS_JSONL).exists()
    assert (out / "decisions_manifest.json").exists()


def test_a_costing_that_does_not_cover_the_arms_is_explained(fx, trained, capsys):
    """R1 refuses to substitute another model's coefficients, which is right.
    It has to reach the operator as a sentence, not a KeyError traceback."""
    argv = [
        "decide", "--root", str(fx.root), "--run", fx.run_id,
        "--tasks", str(fx.tasks_path), "--policy", str(trained),
        "--coefficients", "bench/cost_coefficients.json",
    ]
    assert main(argv) == EXIT_ERROR
    assert "does not cover" in capsys.readouterr().err


def test_decide_refuses_a_d1_policy(fx, tmp_path, capsys):
    """Escalation cannot be replayed as a routing action without dropping the
    cheap arm's already-spent cost."""
    policy = tmp_path / "d1"
    main(["train", "--root", str(fx.root), "--run", fx.run_id,
          "--tasks", str(fx.tasks_path), "--decision-point", "D1",
          "--out", str(policy)])
    assert main(decide_argv(fx, policy)) == EXIT_ERROR
    assert "already spent" in capsys.readouterr().err


# -- repair ------------------------------------------------------------------


@pytest.fixture(scope="module")
def laddered(tmp_path_factory) -> fixtures.Fixture:
    return fixtures.write_fixture(tmp_path_factory.mktemp("ladder_cli"),
                                  SynthConfig(n_tasks=300, seeds=3),
                                  with_ladder=True)


def test_repair_reports_all_three_strategies(laddered, capsys):
    argv = ["repair", "--root", str(laddered.root), "--run", laddered.run_id,
            "--tasks", str(laddered.tasks_path)]
    assert main(argv) == EXIT_OK
    out = capsys.readouterr().out
    for name in ("always_small", "repair", "escalate"):
        assert name in out
    assert "efficiency" in out


def test_repair_on_a_store_with_none_does_not_exit_zero(fx, capsys):
    """NO REPAIRS is not a pass, and it is the state every real store is in."""
    argv = ["repair", "--root", str(fx.root), "--run", fx.run_id,
            "--tasks", str(fx.tasks_path)]
    assert main(argv) == EXIT_GATE_FAILED
    assert "NO REPAIRS" in capsys.readouterr().out


def test_repair_writes_its_verdict(laddered, tmp_path):
    out = tmp_path / "repair.json"
    argv = ["repair", "--root", str(laddered.root), "--run", laddered.run_id,
            "--tasks", str(laddered.tasks_path), "--out", str(out)]
    assert main(argv) == EXIT_OK
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "PASS"
    assert payload["n_repairs"] > 0


def test_the_fixture_command_can_write_a_ladder(tmp_path, capsys):
    assert main(["fixture", "--out", str(tmp_path), "--tasks", "60",
                 "--seeds", "2", "--with-ladder"]) == EXIT_OK
    run_id = capsys.readouterr().out.split("run_id")[1].split()[0]
    rows = (tmp_path / run_id / "generations").glob("part-*.jsonl")
    text = next(rows).read_text()
    assert '"ladder_step": 1' in text


# -- ablate ------------------------------------------------------------------


def test_ablate_reports_a_corrected_family(fx, capsys):
    argv = ["ablate", "--root", str(fx.root), "--run", fx.run_id,
            "--tasks", str(fx.tasks_path), "--resamples", "150"]
    assert main(argv) == EXIT_OK
    out = capsys.readouterr().out
    assert "simultaneous at" in out
    assert "without visible_outcome" in out


def test_ablate_always_exits_zero(fx, capsys):
    """A measurement, not a gate. "No group is detectable" is a real answer and
    wiring it to a non-zero exit would turn a finding into a build failure."""
    argv = ["ablate", "--root", str(fx.root), "--run", fx.run_id,
            "--tasks", str(fx.tasks_path), "--decision-point", "D0",
            "--resamples", "150"]
    assert main(argv) == EXIT_OK


def test_ablate_writes_its_table(fx, tmp_path):
    out = tmp_path / "ablations.json"
    argv = ["ablate", "--root", str(fx.root), "--run", fx.run_id,
            "--tasks", str(fx.tasks_path), "--resamples", "150",
            "--no-only", "--out", str(out)]
    assert main(argv) == EXIT_OK
    payload = json.loads(out.read_text())
    assert payload["carries_the_gap"] == "visible_outcome"
    assert all(d["kind"] == "leave_one_out" for d in payload["deltas"])


def test_ablate_needs_an_arm_when_a_ladder_makes_the_name_ambiguous(
    laddered, capsys,
):
    """`small` and `repair_small` both match the name heuristic, so the cheap
    arm cannot be inferred. Refusing beats guessing — a wrong cheap arm inverts
    every number in the table."""
    argv = ["ablate", "--root", str(laddered.root), "--run", laddered.run_id,
            "--tasks", str(laddered.tasks_path), "--resamples", "80"]
    assert main(argv) == EXIT_ERROR
    assert "cannot tell which" in capsys.readouterr().err

    argv += ["--arm", "small"]
    assert main(argv) == EXIT_OK
