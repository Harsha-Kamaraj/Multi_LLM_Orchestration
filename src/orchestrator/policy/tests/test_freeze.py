"""The freeze, and the ways a freeze can be worthless.

Two failure modes are worth more than the happy path. A freeze over an empty
directory verifies perfectly and guarantees nothing. And a freeze that hashes
only files calls a submission intact after the λ grid moved underneath it,
which changes every number on the frontier without changing a byte.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.policy import decide, fixtures, freeze, heads, store
from orchestrator.policy.freeze import (
    Freeze, FreezeError, freeze_submission, verify,
)
from orchestrator.workers.cost import CostCoefficients
from schemas.synth import SynthConfig

CONFIG = SynthConfig(n_tasks=300, seeds=3)


@pytest.fixture(scope="module")
def submission(tmp_path_factory):
    """A real submission directory: heads, calibration, spec, decisions."""
    root = tmp_path_factory.mktemp("freeze")
    fx = fixtures.write_fixture(root, CONFIG)
    data = store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path,
                               cost_fingerprint=fx.cost_fingerprint)
    policy = heads.fit_heads(data, "D0")

    directory = root / "submission"
    policy.save(directory)
    sweep = decide.sweep_lambda(
        policy, data, CostCoefficients.load(fx.coefficients_path), split="val",
    )
    decide.write_decisions(sweep, directory)
    return directory, data


def test_a_freeze_covers_every_artifact(submission):
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0")
    for name in ("policy.pkl", "calibration.json", "feature_spec.json",
                 "decisions.jsonl"):
        assert frozen.files[name], name


def test_a_freeze_over_nothing_is_refused(tmp_path):
    """The worst failure for an artifact whose whole job is to be checked."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FreezeError, match="no policy to freeze"):
        freeze_submission(tmp_path / "empty", run_id="r", decision_point="D0")


def test_an_untouched_submission_verifies(submission):
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0")
    check = verify(frozen, directory)
    assert check.intact
    assert "INTACT" in check.summary()


def test_editing_an_artifact_breaks_the_freeze(submission):
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0")

    path = directory / "calibration.json"
    original = path.read_text()
    try:
        payload = json.loads(original)
        arm = next(iter(payload))
        payload[arm]["knots_y"] = [0.0, 1.0]
        payload[arm]["knots_x"] = [0.0, 1.0]
        path.write_text(json.dumps(payload))

        check = verify(frozen, directory)
        assert not check.intact
        assert "calibration.json" in check.changed
        assert "BROKEN" in check.summary()
    finally:
        path.write_text(original)


def test_a_deleted_artifact_is_reported_as_missing(submission, tmp_path):
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0")

    partial = tmp_path / "partial"
    partial.mkdir()
    for name in freeze.PINNED_FILES:
        source = directory / name
        if source.is_file() and name != "decisions.jsonl":
            (partial / name).write_bytes(source.read_bytes())

    check = verify(frozen, partial)
    assert "decisions.jsonl" in check.missing
    assert not check.intact


def test_a_file_appearing_after_the_freeze_is_reported(submission, tmp_path):
    """Not obviously wrong, and not obviously fine either — a decisions file
    that did not exist at freeze time was produced with the split open."""
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0")
    thinner = Freeze(
        run_id=frozen.run_id, decision_point=frozen.decision_point,
        created_at=frozen.created_at,
        files={**frozen.files, "decisions.jsonl": None},
        constants=frozen.constants,
    )
    check = verify(thinner, directory)
    assert "decisions.jsonl" in check.appeared


def test_moving_the_lambda_grid_breaks_the_freeze_without_touching_a_file(
    submission,
):
    """The check a bytes-only freeze would miss entirely.

    The same `policy.pkl` swept over a different λ grid produces a different
    frontier, and every file hash still matches.
    """
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0")
    tampered = Freeze(
        run_id=frozen.run_id, decision_point=frozen.decision_point,
        created_at=frozen.created_at, files=frozen.files,
        constants={**frozen.constants, "n_lambdas": 7},
    )
    check = verify(tampered, directory)
    assert not check.intact
    assert "n_lambdas" in check.constants_changed
    assert not check.changed, "no file was touched"


def test_the_thresholds_and_ece_target_are_pinned(submission):
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0")
    assert frozen.constants["gate_thresholds"] == {"D0": 0.65, "D1": 0.75}
    assert frozen.constants["ece_target"] == 0.05


def test_a_freeze_round_trips_through_disk(submission, tmp_path):
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0", note="phase 8")
    reloaded = Freeze.load(frozen.save(tmp_path / "FREEZE.json"))
    assert reloaded == frozen
    assert verify(reloaded, directory).intact


def test_publishability_is_recorded(submission):
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0",
                               publishable=data.publishable)
    assert frozen.publishable == data.publishable


def test_verifying_a_directory_that_is_gone_says_so(submission, tmp_path):
    directory, data = submission
    frozen = freeze_submission(directory, run_id=data.run_id,
                               decision_point="D0")
    with pytest.raises(FreezeError, match="no submission directory"):
        verify(frozen, tmp_path / "nowhere")
