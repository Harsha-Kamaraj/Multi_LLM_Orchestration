"""Every adversarial fixture must be refused at the boundary.

Refusing to produce a number is the correct outcome for all of these. A loader
that limps past a corrupted store is worse than one that raises, because only
the former reaches a report.
"""

from __future__ import annotations

import pytest

from eval import StoreError, TestSplitError, load_rows, unlock_test_split
from eval.loading import TestSplitUnlock
from schemas.adversarial import CASES


def test_clean_store_loads(store):
    assert len(store) > 0
    assert store.arms == ("large", "small")
    assert store.n_seeds == 3
    assert store.splits == ("train", "val")


def test_mixed_versions_refused():
    rows, _ = CASES["mixed_schema_versions"]()
    with pytest.raises(Exception, match="mixes schema versions"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_duplicates_refused():
    rows, _ = CASES["duplicate_rollout_ids"]()
    with pytest.raises(StoreError, match="duplicate rollout_ids"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_split_leakage_refused():
    rows, _ = CASES["split_leakage"]()
    with pytest.raises(StoreError, match="more than one split"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_ungraded_refused_by_default():
    rows, _ = CASES["ungraded_rows_as_zero"]()
    with pytest.raises(StoreError, match="ungraded"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_ungraded_never_silently_becomes_zero():
    """The dangerous half of that fixture: opting out must not turn nulls into
    zeros, which would report a catastrophic accuracy drop as a model result."""
    import numpy as np

    rows, _ = CASES["ungraded_rows_as_zero"]()
    loaded = load_rows(
        rows, splits=("train", "val", "test"), unlock=_unlock(),
        require_graded=False, require_complete_grid=False,
    )
    assert np.isnan(loaded.solved).any(), "ungraded rows must stay NaN, not 0.0"


def test_half_graded_pair_refused():
    rows, _ = CASES["half_graded_pair"]()
    with pytest.raises(StoreError, match="contract"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_single_arm_refused():
    """Structurally valid rows, and still unusable: every comparison here is
    paired across arms."""
    rows, _ = CASES["single_arm_only"]()
    with pytest.raises(StoreError, match="paired across arms"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_incomplete_grid_refused():
    rows, _ = CASES["unsealed_run"]()
    with pytest.raises(StoreError, match="incomplete grid|paired across arms"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_dirty_run_refused_by_default():
    rows, _ = CASES["dirty_run_id"]()
    with pytest.raises(StoreError, match="dirty worktree"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_dirty_run_readable_when_explicit():
    """Local iteration must stay possible; only *publishing* is blocked."""
    rows, _ = CASES["dirty_run_id"]()
    loaded = load_rows(rows, splits=("train", "val", "test"),
                       unlock=_unlock(), allow_dirty=True)
    assert loaded.run_id.endswith("-dirty")


def test_batched_serving_row_refused():
    rows, _ = CASES["serving_row_batched"]()
    with pytest.raises(StoreError, match="contract"):
        load_rows(rows, splits=("train", "val", "test"), unlock=_unlock())


def test_multiple_run_ids_refused(synth):
    rows = [dict(r) for r in synth.rows]
    rows[0]["run_id"] = "2026-02-02-1234567-abcdef"
    with pytest.raises(StoreError, match="spans 2 run_ids|pin exactly one"):
        load_rows(rows)


# --- the frozen split ------------------------------------------------------

def test_test_split_is_locked_by_default(synth):
    with pytest.raises(TestSplitError, match="unlock_test_split"):
        load_rows(synth.rows, splits=("test",))


def test_unlock_requires_a_reason():
    with pytest.raises(TestSplitError, match="reason"):
        TestSplitUnlock(reason="  ", preregistration="x")


def test_unlock_requires_a_preregistration():
    with pytest.raises(TestSplitError, match="pre-registered|preregistration"):
        TestSplitUnlock(reason="final analysis", preregistration="")


def test_unlock_requires_the_prereg_file_to_exist(tmp_path):
    """Pre-registering after unblinding is not pre-registering."""
    with pytest.raises(TestSplitError, match="does not exist"):
        unlock_test_split(reason="final", preregistration=tmp_path / "nope.md")


def test_unlock_succeeds_with_a_real_prereg(tmp_path, synth):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("metric: solved; test: mcnemar; correction: BH\n")
    token = unlock_test_split(reason="final analysis", preregistration=prereg)
    loaded = load_rows(synth.rows, splits=("test",), unlock=token)
    assert loaded.splits == ("test",)


# --- warnings, not failures ------------------------------------------------

def test_hack_flags_surface_as_a_warning(synth):
    rows = [dict(r) for r in synth.rows]
    for row in rows[:20]:
        row["hack_flags"] = ["hardcoded_return"]
    loaded = load_rows(rows)
    assert any("reward-hack" in w for w in loaded.warnings)


def test_truncation_rate_surfaces_as_a_warning(synth):
    rows = [dict(r) for r in synth.rows]
    for row in rows[: len(rows) // 4]:
        row["finish_reason"] = "length"
    loaded = load_rows(rows)
    assert any("truncation rate" in w for w in loaded.warnings)


def test_arm_matrix_is_task_aligned(store):
    """Misaligned task order between arms is a bug that produces a plausible
    number and no error, so the alignment is built once, here."""
    import numpy as np

    solved = store.arm_matrix("_solved")
    assert set(solved) == set(store.arms)
    for arm in store.arms:
        assert solved[arm].shape == (store.n_tasks, store.n_seeds)
        assert not np.isnan(solved[arm]).any()


def test_ordered_tasks_is_stable(store):
    assert store.ordered_tasks == sorted(store.ordered_tasks)
    assert len(store.ordered_tasks) == store.n_tasks


def _unlock() -> TestSplitUnlock:
    return TestSplitUnlock(reason="test suite", preregistration=__file__)
