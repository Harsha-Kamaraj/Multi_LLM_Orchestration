"""Reading R2's graded layer, which is where R3's labels actually live.

A run directory has two layers, sealed independently by different roles:

    runs/{run_id}/generations/  R1's, ungraded, _MANIFEST.json
    runs/{run_id}/rollouts/     R2's, graded,   _ROLLOUT_MANIFEST.json

R2 grades by reading R1's sealed generations and writing
`{**generation, **grade}` into its own directory — never editing R1's files,
because a sealed manifest carries per-file checksums.

The bug these tests exist to prevent is total and silent: a reader that knows
only about `generations/` finds every grading column null, concludes the run
was never graded, and refuses to train — while the labels sit one directory
over. It would look exactly like waiting on R2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.policy import fixtures, store
from orchestrator.policy.errors import StoreReadError, UngradedRunError
from schemas.synth import SynthConfig

CONFIG = SynthConfig(n_tasks=80, seeds=2)


@pytest.fixture
def split_run(tmp_path: Path) -> fixtures.Fixture:
    """The real production shape: ungraded below, graded above."""
    return fixtures.write_fixture(tmp_path, CONFIG, layout="split")


# -- the layout is what R2 actually writes -----------------------------------


def test_the_fixture_reproduces_r2s_two_layer_layout(split_run):
    directory = split_run.root / split_run.run_id
    assert (directory / "generations" / "part-000001-0000000000-5ynth1.jsonl").exists()
    assert (directory / "rollouts" / "part-000001-0000000000-5ynth1.jsonl").exists()
    assert (directory / "_MANIFEST.json").exists()
    assert (directory / "_ROLLOUT_MANIFEST.json").exists()


def test_the_generations_layer_is_ungraded_as_r1_writes_it(split_run):
    data = store.load_rollouts(split_run.root, split_run.run_id,
                               layer="generations", require_grades=False)
    assert data.layer == "generations"
    assert data.labels == {}
    assert all(row["visible_total"] is None for row in data.rows)


# -- the bug this file exists for --------------------------------------------


def test_labels_are_found_in_the_graded_layer(split_run):
    """Without layer selection this run reads as never graded."""
    data = store.load_rollouts(split_run.root, split_run.run_id,
                               tasks_path=split_run.tasks_path)
    assert data.layer == "rollouts"
    assert data.is_graded
    assert len(data.labels) == len(data)


def test_the_graded_layer_is_preferred_automatically(split_run):
    """Naming a layer must not be something a caller has to remember."""
    data = store.load_rollouts(split_run.root, split_run.run_id)
    assert data.layer == "rollouts"


def test_reading_the_ungraded_layer_of_a_graded_run_refuses_to_train(split_run):
    """Asking for generations explicitly gets generations, and they carry no labels."""
    with pytest.raises(UngradedRunError):
        store.load_rollouts(split_run.root, split_run.run_id,
                            layer="generations")


def test_a_run_that_was_never_graded_still_says_so(tmp_path):
    """The message must stay right for the case it was written for."""
    fx = fixtures.write_fixture(tmp_path, CONFIG, layout="split",
                                seal_graded=True)
    # Blank the grades in the graded layer, as an ungraded sweep would have.
    part = next((fx.root / fx.run_id / "rollouts").glob("part-*.jsonl"))
    import json
    rows = [json.loads(line) for line in part.read_text().splitlines() if line]
    for row in rows:
        row["hidden_passed"] = row["hidden_total"] = None
        row["visible_passed"] = row["visible_total"] = None
    part.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    with pytest.raises(UngradedRunError, match="no hidden-test outcome"):
        store.load_rollouts(fx.root, fx.run_id)


# -- interrupted grading is not the same as ungraded -------------------------


def test_an_unsealed_graded_layer_is_refused_rather_than_skipped(tmp_path):
    """Half-graded and never-graded call for completely different actions.

    Falling back to `generations/` here would report a partially graded run as
    never graded, and the operator would go and wait for R2 rather than
    re-running an interrupted grading pass.
    """
    fx = fixtures.write_fixture(tmp_path, CONFIG, layout="split",
                                seal_graded=False)
    with pytest.raises(StoreReadError, match="grading is in progress"):
        store.load_rollouts(fx.root, fx.run_id)


def test_the_ungraded_layer_stays_inspectable_when_grading_is_interrupted(tmp_path):
    fx = fixtures.write_fixture(tmp_path, CONFIG, layout="split",
                                seal_graded=False)
    data = store.load_rollouts(fx.root, fx.run_id, layer="generations",
                               require_grades=False)
    assert len(data) > 0


def test_an_unknown_layer_name_is_refused(split_run):
    with pytest.raises(StoreReadError, match="unknown layer"):
        store.load_rollouts(split_run.root, split_run.run_id, layer="grades")


def test_a_run_with_no_sealed_layer_at_all_is_refused(tmp_path):
    fx = fixtures.write_fixture(tmp_path, CONFIG, sealed=False)
    with pytest.raises(StoreReadError):
        store.load_rollouts(fx.root, fx.run_id)


# -- publishability survives the layer change --------------------------------


def test_publishability_comes_from_r1s_manifest_not_r2s(split_run):
    """R2's manifest has no `publishable` key, and must not default to False."""
    data = store.load_rollouts(split_run.root, split_run.run_id)
    assert data.layer == "rollouts"
    assert "publishable" not in data.manifest
    assert data.generations_manifest["publishable"] is True
    assert data.publishable is True


def test_a_dirty_run_id_still_wins_over_both_manifests(tmp_path):
    fx = fixtures.write_fixture(tmp_path, CONFIG, layout="split")
    directory = fx.root / fx.run_id
    dirty = f"{fx.run_id}-dirty"
    directory.rename(fx.root / dirty)

    data = store.load_rollouts(fx.root, dirty)
    assert data.publishable is False


# -- everything else keeps working on the graded layer -----------------------


def test_the_cost_sidecar_still_joins(split_run):
    data = store.load_rollouts(split_run.root, split_run.run_id,
                               cost_fingerprint=split_run.cost_fingerprint)
    assert data.has_cost
    assert all(row["gpu_seconds"] is not None for row in data.rows)


def test_hidden_columns_are_stripped_from_the_graded_layer_too(split_run):
    """The strip runs on whichever layer supplied the rows."""
    data = store.load_rollouts(split_run.root, split_run.run_id)
    for row in data.rows:
        assert "hidden_passed" not in row
        assert "hidden_total" not in row


def test_the_gate_runs_against_the_graded_layer(split_run):
    from orchestrator.policy import gate

    data = store.load_rollouts(split_run.root, split_run.run_id,
                               tasks_path=split_run.tasks_path,
                               cost_fingerprint=split_run.cost_fingerprint)
    result = gate.measure_gate(data, "D1", n_resamples=50)
    assert result.auc.point > 0.5


def test_parquet_and_jsonl_agree_on_the_graded_layer(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    import json

    fx = fixtures.write_fixture(tmp_path, CONFIG, layout="split")
    rows_dir = fx.root / fx.run_id / "rollouts"
    part = next(rows_dir.glob("part-*.jsonl"))
    rows = [json.loads(line) for line in part.read_text().splitlines() if line]
    for row in rows:
        row["extra"] = json.dumps(row.get("extra") or {}, sort_keys=True)
    pq.write_table(pa.Table.from_pylist(rows), rows_dir / "rollouts.parquet")

    via_parquet = store.load_rollouts(fx.root, fx.run_id, prefer_parquet=True)
    via_jsonl = store.load_rollouts(fx.root, fx.run_id, prefer_parquet=False)
    assert via_parquet.source == "parquet"
    assert list(via_parquet.rows) == list(via_jsonl.rows)


# -- drift canary ------------------------------------------------------------


def test_the_graded_layer_constants_match_r2s_module():
    """Fails loudly when R2's branch lands and the names disagree.

    `graders.rollout_store` is not on `main` yet, so `store.GRADED_LAYER`
    restates its constants. This test skips until the module exists and then
    starts enforcing agreement — which is the moment the restatement becomes
    dangerous rather than necessary.
    """
    r2 = pytest.importorskip(
        "orchestrator.graders.rollout_store",
        reason="R2's grading store has not landed on main yet",
    )
    assert store.GRADED_LAYER.rows_dir == r2.ROWS_DIR
    assert store.GRADED_LAYER.manifest_name == r2.MANIFEST_NAME
    assert store.GRADED_LAYER.parquet_name == r2.PARQUET_NAME
