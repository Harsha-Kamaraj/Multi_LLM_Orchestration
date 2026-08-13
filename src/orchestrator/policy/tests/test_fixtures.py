"""The synthetic fixture, read through the same path production will use.

Every assertion here goes through `load_rollouts` rather than through
`SynthResult.rows`. That is the point of the adapter: if a test can only pass
by reading a list of dicts the loader would have rejected, it is testing
something the real pipeline never does.

The signal test at the bottom is the Phase 2 deliverable — the planted D0
signal must be recoverable through a *legitimate* channel, with latent
difficulty quarantined. If it were only recoverable from `.latent`, the fixture
would be unusable for the thing it exists for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orchestrator.policy import fixtures, store
from orchestrator.policy.errors import StoreReadError
from schemas.synth import SynthConfig


@pytest.fixture(scope="module")
def config() -> SynthConfig:
    # Big enough that an AUC over it is not noise, small enough to stay fast.
    return SynthConfig(n_tasks=200, seeds=3)


@pytest.fixture
def fx(tmp_path: Path, config: SynthConfig) -> fixtures.Fixture:
    return fixtures.write_fixture(tmp_path, config)


@pytest.fixture
def loaded(fx: fixtures.Fixture) -> store.RolloutData:
    return store.load_rollouts(
        fx.root, fx.run_id,
        tasks_path=fx.tasks_path,
        cost_fingerprint=fx.cost_fingerprint,
    )


# -- it is a real run directory ----------------------------------------------


def test_the_fixture_loads_through_the_production_read_path(loaded, config):
    # 200 tasks x 3 seeds x 2 arms, less the 20% held in test.
    assert 0 < len(loaded) < config.n_tasks * config.seeds * 2
    assert sorted(loaded.arms) == ["large", "small"]
    assert loaded.source == "jsonl"


def test_the_test_split_is_absent_from_a_loaded_fixture(loaded):
    assert set(loaded.counts_by_split()) <= {"train", "val"}


def test_an_unsealed_fixture_is_skipped_like_an_interrupted_sweep(
    tmp_path: Path, config: SynthConfig,
):
    fx = fixtures.write_fixture(tmp_path, config, sealed=False)
    with pytest.raises(StoreReadError, match="_MANIFEST"):
        store.load_rollouts(fx.root, fx.run_id)


def test_a_fixture_without_a_costing_attaches_none(tmp_path: Path,
                                                   config: SynthConfig):
    fx = fixtures.write_fixture(tmp_path, config, with_cost=False)
    assert fx.cost_fingerprint is None
    data = store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path)
    assert data.has_cost is False


def test_the_pinned_costing_agrees_with_the_rows_it_was_built_from(fx):
    """A disagreement here is a bug in the fixture wearing the join's clothes."""
    unpriced = store.load_rollouts(fx.root, fx.run_id)
    priced = store.load_rollouts(fx.root, fx.run_id,
                                 cost_fingerprint=fx.cost_fingerprint)
    by_id = {row["rollout_id"]: row for row in unpriced.rows}
    for row in priced.rows:
        assert row["gpu_seconds"] == by_id[row["rollout_id"]]["gpu_seconds"]


# -- the three-way separation ------------------------------------------------


def test_labels_latent_and_rows_are_three_separate_things(loaded):
    assert len(loaded.labels) == len(loaded)
    assert len(loaded.latent) == len(loaded)
    for row in loaded.rows:
        assert not any(k.startswith("hidden") for k in row)
        assert not any(k.startswith("_synth") for k in row.get("extra", {}))


def test_latent_difficulty_is_reachable_only_from_latent(loaded):
    """It is ground truth. A test may see it; a feature builder may not."""
    assert loaded.is_synthetic
    planted = next(iter(loaded.latent.values()))
    assert "difficulty" in planted
    assert all("difficulty" not in row for row in loaded.rows)


# -- the D0 channel ----------------------------------------------------------


def test_the_proxy_arrives_through_the_prompt_side_join(loaded, fx):
    """The rollout row carries no prompt, so D0 features come from the corpus."""
    truth = {t: float(x) for t, x in
             zip(fx.truth["task_ids"], fx.truth["x_d0"])}
    for row in loaded.rows:
        assert row["task_x_d0"] == pytest.approx(truth[row["task_id"]])


def test_no_proxy_column_appears_without_the_join(fx):
    data = store.load_rollouts(fx.root, fx.run_id)
    assert all("task_x_d0" not in row for row in data.rows)


def test_a_real_corpus_produces_no_proxy_column(tmp_path: Path, fx):
    """`task_x_d0` is fixture-only, and must not be defaulted into existence."""
    corpus = tmp_path / "real_tasks.jsonl"
    import json
    with corpus.open("w", encoding="utf-8") as fh:
        for task_id in fx.truth["task_ids"]:
            fh.write(json.dumps({
                "task_id": task_id, "prompt": "Write a function.",
                "entrypoint": "solve", "visible_tests": "assert solve()",
            }) + "\n")

    data = store.load_rollouts(fx.root, fx.run_id, tasks_path=corpus)
    assert all("task_x_d0" not in row for row in data.rows)
    assert all("task_prompt" in row for row in data.rows)


def test_prompt_length_tracks_the_proxy_inversely(loaded):
    """So a genuine prompt-length feature is a valid D0 signal on the fixture.

    Negative by design: `x_d0` is oriented so higher means more likely to
    solve, and a longer prompt means a harder task. Planting the relationship
    in the other direction would validate a feature builder against a sign the
    real store does not have.
    """
    lengths = np.array([len(row["task_prompt"].split()) for row in loaded.rows])
    proxy = np.array([row["task_x_d0"] for row in loaded.rows])
    assert np.corrcoef(lengths, proxy)[0, 1] < -0.99


# -- the deliverable ---------------------------------------------------------


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC with ties averaged. Mirrors the generator's own definition."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def test_the_planted_signal_survives_the_legitimate_channel(loaded):
    """Phase 2's deliverable: recoverable without touching ground truth.

    Deliberately not a model — a single feature and a rank statistic. If the
    planted signal needs a fitted model to show up at all, then a Phase 4 AUC
    is measuring the model rather than the data, and the fixture has stopped
    being a correctness test.
    """
    small = [row for row in loaded.rows if row["arm"] == "small"]
    proxy = np.array([row["task_x_d0"] for row in small])
    solved = np.array([loaded.label_for(row["rollout_id"]).solved
                       for row in small])

    auc = _auc(proxy, solved)
    assert 0.60 < auc < 0.80, (
        f"planted D0 signal came back at AUC {auc:.3f}. Below the band the "
        f"fixture is too noisy to validate against; above it, something more "
        f"informative than a prompt proxy has reached the feature path."
    )


def test_prompt_length_alone_recovers_the_signal(loaded):
    """The end-to-end version: no fixture-specific column involved at all."""
    small = [row for row in loaded.rows if row["arm"] == "small"]
    lengths = np.array([len(row["task_prompt"].split()) for row in small])
    solved = np.array([loaded.label_for(row["rollout_id"]).solved
                       for row in small])
    # Negated because a longer prompt means a harder task.
    assert _auc(-lengths, solved) > 0.60


def test_the_large_arm_solves_a_correlated_superset(loaded):
    """The structure that makes routing a real question rather than a coin toss."""
    by_arm: dict[str, list[bool]] = {"small": [], "large": []}
    for row in loaded.rows:
        by_arm[row["arm"]].append(loaded.label_for(row["rollout_id"]).solved)

    small_rate = float(np.mean(by_arm["small"]))
    large_rate = float(np.mean(by_arm["large"]))
    # Phase 0's gate is an 8pp gap between arms; the fixture plants ~20pp.
    assert large_rate - small_rate > 0.08
