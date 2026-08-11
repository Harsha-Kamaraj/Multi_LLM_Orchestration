"""Arm registry invariants."""

from __future__ import annotations

import pytest

from orchestrator.workers.arms import (
    ARMS, DEFAULT_SEEDS, FROZEN_LADDER, Arm, get_arm, resolve_arms,
)
from orchestrator.workers.params import GREEDY


def test_frozen_ladder_is_the_phase_1_pair():
    """Six generations per task: two arms, three seeds."""
    assert FROZEN_LADDER == ("direct_small", "direct_large")
    assert len(DEFAULT_SEEDS) == 3


def test_arms_name_roles_not_models():
    """Swapping which weights back a rung is a backend argument, not an edit
    to the arm registry."""
    for arm in ARMS.values():
        assert arm.model_role in ("small", "large")


def test_unknown_arm_raises_rather_than_defaulting():
    """A fallback would generate under a different configuration than the one
    recorded on the row."""
    with pytest.raises(KeyError, match="unknown arm"):
        get_arm("direct_medium")


def test_resolve_defaults_to_the_frozen_ladder():
    assert [a.name for a in resolve_arms(None)] == list(FROZEN_LADDER)


def test_the_two_direct_arms_share_sampling_and_differ_only_in_rung():
    """Same params_hash is correct — they differ by model, and the model comes
    from the arm, which is already in the resume key."""
    small, large = get_arm("direct_small"), get_arm("direct_large")
    assert small.params_hash == large.params_hash
    assert small.model_role != large.model_role


def test_the_ablation_arms_hash_differently():
    assert get_arm("direct_small_notests").params_hash != get_arm("direct_small").params_hash


def test_the_probe_arm_samples():
    """k identical samples carry no self-consistency signal."""
    assert get_arm("probe_small").params.temperature > 0


def test_repair_arms_sit_above_step_zero_and_need_a_parent():
    for name in ("repair_small", "repair_large"):
        arm = get_arm(name)
        assert arm.ladder_step > 0 and arm.requires_parent


def test_a_repair_arm_without_a_parent_is_rejected():
    """A repair with no parent would generate from scratch and be recorded as
    a repair, making the ladder look ineffective for the wrong reason."""
    with pytest.raises(ValueError, match="not a repair"):
        Arm(name="bad", model_role="small", params=GREEDY, ladder_step=1)


def test_a_step_zero_arm_requiring_a_parent_is_rejected():
    with pytest.raises(ValueError, match="ladder step 0"):
        Arm(name="bad", model_role="small", params=GREEDY, requires_parent=True)


def test_every_registered_arm_is_self_consistent():
    for name, arm in ARMS.items():
        assert arm.name == name
        assert arm.description
