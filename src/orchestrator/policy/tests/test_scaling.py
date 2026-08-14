"""Normalization discipline: fitted on train, serialized, never refitted.

The tests that matter here are the negative ones. Standardizing correctly and
standardizing leakily produce matrices that look the same and differ only in
how good the eventual model looks, so the guards are the point.
"""

from __future__ import annotations

import numpy as np
import pytest

from orchestrator.policy import fixtures, store
from orchestrator.policy.errors import SplitError
from orchestrator.policy.features import FeatureError
from orchestrator.policy.features.d0 import D0_FEATURES
from orchestrator.policy.features.scaling import (
    Standardizer, fit_standardizer,
)
from schemas.synth import SynthConfig


@pytest.fixture(scope="module")
def loaded(tmp_path_factory) -> store.RolloutData:
    root = tmp_path_factory.mktemp("scaling")
    fx = fixtures.write_fixture(root, SynthConfig(n_tasks=150, seeds=2))
    return store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path)


@pytest.fixture(scope="module")
def matrix(loaded):
    return D0_FEATURES.build(loaded.rows)


# -- the discipline ----------------------------------------------------------


def test_constants_come_from_train_only(loaded, matrix):
    """The whole point: val rows must not move the mean."""
    scaler = fit_standardizer(matrix, loaded.rows)

    train_mask = np.array([r["split"] == "train" for r in loaded.rows])
    expected = matrix.X[train_mask].mean(axis=0)

    np.testing.assert_allclose(scaler.mean, expected)
    assert scaler.fitted_on == ("train",)
    assert scaler.n_rows == int(train_mask.sum())


def test_constants_differ_from_the_leaky_version(loaded, matrix):
    """If these agreed, the test above would prove nothing."""
    scaler = fit_standardizer(matrix, loaded.rows)
    leaky = matrix.X.mean(axis=0)
    assert not np.allclose(scaler.mean, leaky)


def test_fitting_on_test_is_refused(loaded, matrix):
    with pytest.raises(SplitError, match="Fitting anything on test"):
        fit_standardizer(matrix, loaded.rows, on=("train", "test"))


def test_fitting_on_no_split_is_refused(loaded, matrix):
    with pytest.raises(SplitError, match="at least one split"):
        fit_standardizer(matrix, loaded.rows, on=())


def test_a_split_with_no_rows_says_what_is_present(loaded, matrix):
    rows = [dict(r, split="train") for r in loaded.rows]
    with pytest.raises(SplitError, match=r"\['train'\]"):
        fit_standardizer(matrix, rows, on=("val",))


def test_misaligned_rows_and_matrix_are_refused(loaded, matrix):
    """A shorter mask would silently select the wrong observations."""
    with pytest.raises(FeatureError, match="same rows in the same order"):
        fit_standardizer(matrix, loaded.rows[:-1])


# -- transform ---------------------------------------------------------------


def test_transform_centres_the_training_split(loaded, matrix):
    scaler = fit_standardizer(matrix, loaded.rows)
    scaled = scaler.transform(matrix)

    train_mask = np.array([r["split"] == "train" for r in loaded.rows])
    varying = [i for i, n in enumerate(scaled.names)
               if n not in scaler.constant_columns]
    block = scaled.X[train_mask][:, varying]
    np.testing.assert_allclose(block.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(block.std(axis=0), 1.0, atol=1e-9)


def test_the_validation_split_is_not_centred_by_construction(loaded, matrix):
    """It is scaled by train's constants, so its own mean is near zero, not at it."""
    scaler = fit_standardizer(matrix, loaded.rows)
    scaled = scaler.transform(matrix)

    val_mask = np.array([r["split"] == "val" for r in loaded.rows])
    varying = [i for i, n in enumerate(scaled.names)
               if n not in scaler.constant_columns]
    val_means = scaled.X[val_mask][:, varying].mean(axis=0)
    assert not np.allclose(val_means, 0.0, atol=1e-12)


def test_a_constant_column_passes_through_rather_than_dividing_by_zero(
    loaded, matrix,
):
    scaler = fit_standardizer(matrix, loaded.rows)
    assert scaler.constant_columns, "fixture should have some constant columns"
    scaled = scaler.transform(matrix)
    assert np.isfinite(scaled.X).all()
    for name in scaler.constant_columns:
        assert np.isfinite(scaled.column(name)).all()


def test_a_constant_column_is_centred_to_zero_not_left_alone(loaded, matrix):
    """`scale=1.0` avoids the division; it does not skip the centring.

    Worth pinning because it is easy to describe as "passed through untouched",
    which is a different behaviour that no test would have caught.
    """
    scaler = fit_standardizer(matrix, loaded.rows)
    scaled = scaler.transform(matrix)
    for name in scaler.constant_columns:
        np.testing.assert_allclose(scaled.column(name), 0.0, atol=1e-12)
        index = scaler.names.index(name)
        assert scaler.scale[index] == 1.0


def test_transform_refuses_a_different_feature_set(loaded, matrix):
    from orchestrator.policy.features.d1 import D1_FEATURES

    scaler = fit_standardizer(matrix, loaded.rows)
    other = D1_FEATURES.build(loaded.rows)
    with pytest.raises(FeatureError, match="different feature set"):
        scaler.transform(other)


def test_transform_refuses_the_right_columns_in_the_wrong_order(loaded, matrix):
    """Reordered columns would be scaled by the wrong constants, silently."""
    from orchestrator.policy.features.spec import FeatureMatrix

    scaler = fit_standardizer(matrix, loaded.rows)
    order = list(range(len(matrix.names)))[::-1]
    shuffled = FeatureMatrix(
        X=matrix.X[:, order],
        names=tuple(matrix.names[i] for i in order),
        rollout_ids=matrix.rollout_ids,
        decision_point=matrix.decision_point,
    )
    with pytest.raises(FeatureError, match="order matters|Order matters"):
        scaler.transform(shuffled)


def test_transform_preserves_the_row_keys(loaded, matrix):
    scaler = fit_standardizer(matrix, loaded.rows)
    scaled = scaler.transform(matrix)
    assert scaled.rollout_ids == matrix.rollout_ids
    assert scaled.decision_point == matrix.decision_point


# -- serialization -----------------------------------------------------------


def test_a_standardizer_round_trips_through_disk(tmp_path, loaded, matrix):
    """It ships beside the model, so a later scoring run reuses, never refits."""
    scaler = fit_standardizer(matrix, loaded.rows)
    reloaded = Standardizer.load(scaler.save(tmp_path / "scaling.json"))

    assert reloaded == scaler
    np.testing.assert_array_equal(
        reloaded.transform(matrix).X, scaler.transform(matrix).X
    )


def test_the_serialized_form_records_which_split_it_saw(tmp_path, loaded, matrix):
    import json

    scaler = fit_standardizer(matrix, loaded.rows)
    payload = json.loads(scaler.save(tmp_path / "scaling.json").read_text())
    assert payload["fitted_on"] == ["train"]
    assert payload["n_rows"] > 0


def test_mismatched_lengths_are_refused_at_construction():
    with pytest.raises(FeatureError, match="must agree"):
        Standardizer(names=("a", "b"), mean=(0.0,), scale=(1.0, 1.0),
                     fitted_on=("train",), n_rows=10)
