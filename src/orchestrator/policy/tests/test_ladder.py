"""The repair ladder, its causality guard, and whether repair pays for itself.

The guard is the point of this file. `docs/harsha.md` lists five leaks that
actually happen, and "a feature from ladder step k+1 used at step k" was the
only one that could not be closed before a ladder existed. It closes here, and
it closes structurally: the later steps are *absent* from what a step-k caller
holds, rather than present and forbidden.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from orchestrator.policy import fixtures, ladder, store
from orchestrator.policy.errors import LeakageError
from orchestrator.policy.ladder import (
    Ladder, LadderError, build_ladders, measure_repair_gate,
)
from schemas.synth import SynthConfig

CONFIG = SynthConfig(n_tasks=400, seeds=3)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return fixtures.write_fixture(tmp_path_factory.mktemp("ladder"), CONFIG,
                                  with_ladder=True)


@pytest.fixture(scope="module")
def loaded(fx) -> store.RolloutData:
    return store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path)


@pytest.fixture(scope="module")
def ladders(loaded):
    return build_ladders(loaded.rows)


def _repaired(ladders) -> Ladder:
    return next(l for l in ladders.values() if len(l) > 1)


# -- assembly ---------------------------------------------------------------


def test_the_fixture_produces_repairs(ladders):
    assert any(len(l) > 1 for l in ladders.values())


def test_a_ladder_is_keyed_by_its_root_not_by_task_and_seed(loaded, ladders):
    """`(task_id, seed)` is shared by every arm's step-0 sample.

    Keying on it merges the cheap and expensive attempts into one impossible
    chain with two roots — which surfaces as "two rows at the same step",
    an error that points at the data instead of at the key.
    """
    for chain in ladders.values():
        assert len(set(chain.steps)) == len(chain.steps)
        assert chain.root.ladder_step == 0
    arms = {l.root_arm for l in ladders.values()}
    assert {"small", "large"} <= arms


def test_every_repair_hangs_off_a_real_parent(ladders):
    for chain in ladders.values():
        for node in chain.nodes[1:]:
            assert node.parent_rollout_id
            assert node.ladder_step > 0


def test_an_orphan_repair_is_refused(loaded):
    """Promoting it to a root would credit a repair's outcome to a direct
    generation while charging none of the parent's cost."""
    orphaned = tuple(
        dict(r, parent_rollout_id="0000000000000000")
        if int(r.get("ladder_step") or 0) > 0 else dict(r)
        for r in loaded.rows
    )
    with pytest.raises(LadderError, match="not in this store"):
        build_ladders(orphaned)


def test_a_root_with_a_parent_is_refused(loaded):
    rows = list(loaded.rows)
    rows[0] = dict(rows[0], ladder_step=0, parent_rollout_id="abc")
    with pytest.raises(LadderError, match="A root has nothing above it"):
        build_ladders(rows)


def test_a_repair_with_no_parent_is_refused(loaded):
    rows = [dict(r) for r in loaded.rows]
    for row in rows:
        if int(row.get("ladder_step") or 0) > 0:
            row["parent_rollout_id"] = None
            break
    with pytest.raises(LadderError, match="nothing says what it was repairing"):
        build_ladders(rows)


def test_a_cycle_is_refused(loaded):
    a, b = [dict(r) for r in loaded.rows[:2]]
    a.update(ladder_step=1, parent_rollout_id=b["rollout_id"])
    b.update(ladder_step=1, parent_rollout_id=a["rollout_id"])
    with pytest.raises(LadderError, match="cycle"):
        build_ladders([a, b])


def test_two_attempts_at_one_step_are_refused(loaded, ladders):
    chain = _repaired(ladders)
    twin = dict(chain.nodes[1].row)
    twin["rollout_id"] = "dup0000000000000"
    with pytest.raises(LadderError, match="two rows at the same ladder step"):
        build_ladders([n.row for n in chain.nodes] + [twin])


# -- the causality guard ----------------------------------------------------


def test_truncation_removes_the_later_steps_entirely(ladders):
    """Structural, not a rule: what is absent cannot be read by accident."""
    chain = _repaired(ladders)
    assert len(chain) == 2
    assert len(chain.upto(0)) == 1
    assert chain.upto(0).steps == (0,)


def test_reading_the_repair_while_deciding_to_repair_is_a_leak(ladders):
    """The fifth named leak, as an exception.

    A feature that reads step 1 while the decision at step 0 is *whether to
    run step 1* has read the answer. It would look like an excellent policy.
    """
    chain = _repaired(ladders)
    with pytest.raises(LeakageError, match="same mistake as reading the hidden"):
        chain.upto(0).at(1)


def test_a_missing_step_and_a_hidden_step_are_different_errors(ladders):
    """"This ladder has no step 2" and "you may not see step 2 yet" are
    different bugs, and a single message would send you after the wrong one."""
    chain = _repaired(ladders)
    with pytest.raises(LadderError, match="has no step 2"):
        chain.at(2)
    with pytest.raises(LeakageError, match="not observable at step 0"):
        chain.upto(0).at(1)


def test_rows_upto_hands_back_only_what_was_observable(ladders):
    chain = _repaired(ladders)
    assert len(chain.rows_upto(0)) == 1
    assert len(chain.rows_upto(1)) == 2


def test_a_negative_step_is_refused(ladders):
    with pytest.raises(LadderError, match="steps start at 0"):
        _repaired(ladders).upto(-1)


# -- cost is cumulative -----------------------------------------------------


def test_a_ladder_costs_every_step_it_took(ladders):
    """Charging only the accepted step is the easiest way to make repair look
    free, and it flatters repair exactly where the comparison matters."""
    chain = _repaired(ladders)
    step0 = float(chain.at(0).row["gpu_seconds"])
    step1 = float(chain.at(1).row["gpu_seconds"])
    assert chain.cumulative_cost(0) == pytest.approx(step0)
    assert chain.cumulative_cost(1) == pytest.approx(step0 + step1)


def test_an_uncosted_step_raises_rather_than_being_skipped(ladders):
    chain = _repaired(ladders)
    blinded = Ladder(
        task_id=chain.task_id, seed=chain.seed,
        nodes=tuple(replace(n, row=dict(n.row, gpu_seconds=None))
                    for n in chain.nodes),
    )
    with pytest.raises(LadderError, match="the ladder's cost is unknown"):
        blinded.cumulative_cost(1)


# -- does repair pay for itself? --------------------------------------------


@pytest.fixture(scope="module")
def gate(loaded):
    return measure_repair_gate(loaded)


def test_the_gate_replays_only_cheap_rooted_ladders(gate, ladders):
    """A chain rooted at the expensive arm is not a decision point — you only
    choose between repair and escalation after the *cheap* attempt failed."""
    cheap_rooted = sum(1 for l in ladders.values() if l.root_arm == "small")
    assert gate.repair.n == cheap_rooted


def test_all_three_strategies_pay_for_the_cheap_attempt(gate):
    """They all take it. Charging repair only for the repair would compare it
    against a baseline that never ran."""
    assert gate.repair.cost > gate.always_small.cost
    assert gate.escalate.cost > gate.always_small.cost


def test_escalation_buys_more_accuracy_and_costs_much_more(gate):
    assert gate.escalate.accuracy > gate.repair.accuracy
    assert gate.escalate.cost > gate.repair.cost


def test_repair_is_the_more_efficient_purchase(gate):
    """ROADMAP's Phase 2 gate, on the fixture's invented repair semantics."""
    assert gate.repair_efficiency > gate.escalation_efficiency
    assert gate.verdict == "PASS"


def test_repair_wins_only_in_a_lambda_region(gate):
    """Not everywhere. Repair is cheaper and weaker, so which one wins is a
    question about price — a gate saying "repair is better" has no answer."""
    assert gate.repair_wins
    assert len(gate.repair_wins) < len(gate.lambdas)


def test_repair_wins_at_high_lambda_and_loses_at_low(gate):
    assert gate.lambdas[-1] in gate.repair_wins
    assert gate.lambdas[0] not in gate.repair_wins


def test_the_gate_serializes(gate):
    payload = gate.as_dict()
    assert payload["verdict"] == "PASS"
    assert set(payload["strategies"]) == {
        "always_small", "repair_on_failure", "escalate_on_failure",
    }


def test_a_store_with_no_repairs_reports_no_repairs_rather_than_passing(tmp_path):
    """The state every real store is in today, and it produced a false PASS.

    With no repair rows the repair strategy silently *becomes* `always_small`,
    which beats escalation at high λ purely by declining to spend. The gate
    would have reported "repair pays for itself" having repaired nothing. It
    has to be named, not scored.
    """
    fx = fixtures.write_fixture(tmp_path, SynthConfig(n_tasks=120, seeds=2))
    data = store.load_rollouts(fx.root, fx.run_id, tasks_path=fx.tasks_path)
    result = measure_repair_gate(data)

    assert result.n_repairs == 0
    assert result.repair.accuracy == result.always_small.accuracy
    assert result.repair.cost == pytest.approx(result.always_small.cost)
    assert result.verdict == "NO REPAIRS"
    assert result.repair_wins, "the degenerate strategy does win on utility"
    assert "under another name" in result.summary()
