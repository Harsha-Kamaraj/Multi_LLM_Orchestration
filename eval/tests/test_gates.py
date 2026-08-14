"""The Phase 0 gate must adjudicate, not editorialize.

The property that matters most: **unmeasured is not passed**. A gate that
quietly skips itself when its inputs are missing is worse than one that fails,
because the phase advances on evidence nobody produced.
"""

from __future__ import annotations

import pytest

from eval.gates import Gate, evaluate


@pytest.fixture(scope="module")
def gates(store):
    return evaluate(store)


def test_all_four_quantities_are_reported(gates):
    names = [g.name for g in gates.gates]
    assert names == ["A_large - A_small", "A_oracle - A_large", "AUC_D0", "AUC_D1"]


def test_only_auc_d1_is_a_hard_stop(gates):
    """The others change the plan rather than ending it. Saying so in advance
    stops a disappointing number being renegotiated into a passing one."""
    hard = [g.name for g in gates.gates if g.hard_stop]
    assert hard == ["AUC_D1"]


def test_unmeasured_is_neither_passed_nor_failed(store):
    """The property the whole module rests on."""
    d0 = next(g for g in evaluate(store).gates if g.name == "AUC_D0")
    assert d0.value is None
    assert d0.passed is None
    assert not d0.measured
    assert "task manifest" in d0.detail


def test_a_partially_measured_gate_does_not_advance_the_phase(gates):
    assert not gates.all_measured
    assert "not a passed gate" in str(gates)


def test_the_fixture_clears_the_arm_gap(gates):
    gap = next(g for g in gates.gates if g.name == "A_large - A_small")
    assert gap.measured and gap.passed
    assert gap.value > 0.08


def test_the_oracle_headroom_gate_adjudicates_rather_than_rubber_stamps(gates):
    """The fixture measures ~4.2pp of headroom against a 5pp threshold, so this
    gate legitimately FAILS on it — and that is the point. A gate that passed
    everything handed to it would be decoration.

    A soft failure here says: the large arm nearly dominates, so only cost
    savings are winnable on this data, not accuracy. That is a real conclusion
    about the fixture, not a defect in it.
    """
    headroom = next(g for g in gates.gates if g.name == "A_oracle - A_large")
    assert headroom.measured
    assert headroom.passed is (headroom.value >= headroom.threshold)
    assert not headroom.hard_stop, "this failure changes the plan, it does not end it"


def test_auc_d1_is_measurable_from_the_store_alone(gates):
    """D1 features are on the row — visible-test outcomes — so this gate never
    needs the corpus join that D0 does."""
    d1 = next(g for g in gates.gates if g.name == "AUC_D1")
    assert d1.measured
    assert d1.value > 0.5


def test_auc_d0_is_measured_when_a_corpus_is_supplied(store):
    corpus = [
        {"task_id": t, "prompt": "x " * (i % 40 + 1),
         "visible_tests": "assert f(1)\n" * (i % 5 + 1), "hidden_tests": ""}
        for i, t in enumerate(store.ordered_tasks)
    ]
    d0 = next(g for g in evaluate(store, corpus=corpus).gates if g.name == "AUC_D0")
    assert d0.measured
    assert 0.4 <= d0.value <= 1.0
    assert "prompt feature" in d0.detail


def test_blocked_only_when_a_hard_stop_fails(gates):
    assert not gates.blocked


def test_a_failed_hard_stop_blocks():
    report_gates = [
        Gate("AUC_D1", 0.51, 0.75, hard_stop=True, on_failure="premise is false")
    ]
    from eval.gates import GateReport

    report = GateReport(gates=report_gates)
    assert report.blocked
    assert "HARD STOP" in str(report)


def test_a_failed_soft_gate_does_not_block():
    from eval.gates import GateReport

    report = GateReport(gates=[
        Gate("A_large - A_small", 0.01, 0.08, hard_stop=False,
             on_failure="shrink the small arm"),
    ])
    assert not report.blocked
    assert report.failures
    assert "shrink the small arm" in str(report)


def test_failure_advice_is_printed_with_the_failure():
    """A gate that says only FAIL makes the reader rediscover what to do."""
    from eval.gates import GateReport

    text = str(GateReport(gates=[
        Gate("A_oracle - A_large", 0.01, 0.05, hard_stop=False,
             on_failure="the large arm dominates; only cost is winnable"),
    ]))
    assert "only cost is winnable" in text


def test_serialization_keeps_unmeasured_distinct_from_failed(gates):
    payload = gates.as_dict()
    d0 = next(g for g in payload["gates"] if g["name"] == "AUC_D0")
    assert d0["passed"] is None
    assert d0["value"] is None
