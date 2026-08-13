"""The decision-point guards, at declaration and at access.

These are the tests that matter most in this package. A leaked feature does not
crash and does not look wrong — it produces a better AUC, which is the one
outcome nobody investigates. So both layers are tested against a feature that
deliberately misbehaves, rather than only against the well-behaved ones in
`d0.py` and `d1.py`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pytest

from orchestrator.policy import contract
from orchestrator.policy.errors import LeakageError
from orchestrator.policy.features import (
    Feature, FeatureError, FeatureSet, RowView, feature, feature_set,
)
from orchestrator.policy.features.d0 import D0_FEATURES
from orchestrator.policy.features.d1 import D1_FEATURES, PROBE_FEATURES


def a_row(**overrides) -> dict:
    row = {
        "rollout_id": "r0",
        "task_id": "synth/00001",
        "arm": "small",
        "seed": 0,
        "task_prompt": "Write a function that sorts a list.",
        "task_entrypoint": "solve",
        "task_visible_tests": "assert solve([2,1]) == [1,2]",
        "code": "def solve(xs):\n    return sorted(xs)\n",
        "code_parses": True,
        "visible_passed": 2,
        "visible_total": 4,
        "finish_reason": "stop",
        "decode_tokens": 120,
        "prefill_tokens": 80,
        "extract_strategy": "fenced",
        "error": None,
        "hack_flags": (),
    }
    row.update(overrides)
    return row


# -- layer one: the declaration is checked against the contract --------------


def test_a_d0_feature_declaring_a_d1_column_is_refused():
    with pytest.raises(LeakageError, match="not observable"):
        Feature(
            name="peeks_ahead",
            decision_point="D0",
            source_columns=("visible_passed",),
            fn=lambda view: 0.0,
        )


def test_the_refusal_explains_that_it_is_really_a_d1_feature():
    with pytest.raises(LeakageError, match="only after generating"):
        Feature(name="peeks_ahead", decision_point="D0",
                source_columns=("code",), fn=lambda view: 0.0)


@pytest.mark.parametrize("column", sorted(contract.LABEL_COLUMNS))
def test_no_feature_may_declare_a_label_at_any_decision_point(column):
    for point in ("D0", "D1"):
        with pytest.raises(LeakageError, match="never features"):
            Feature(name="cheats", decision_point=point,
                    source_columns=(column,), fn=lambda view: 0.0)


def test_sweep_wall_clock_is_refused_at_both_decision_points():
    """It measures queue depth, so a feature on it measures batch composition."""
    for point in ("D0", "D1"):
        with pytest.raises(LeakageError, match="queue-depth"):
            Feature(name="fast", decision_point=point,
                    source_columns=("wall_ms",), fn=lambda view: 0.0)


def test_a_feature_that_declares_nothing_is_refused():
    with pytest.raises(FeatureError, match="constant"):
        Feature(name="empty", decision_point="D0", source_columns=(),
                fn=lambda view: 1.0)


def test_an_unknown_decision_point_is_refused():
    with pytest.raises(FeatureError, match="exactly"):
        Feature(name="x", decision_point="D2", source_columns=("task_prompt",),
                fn=lambda view: 0.0)


# -- layer two: access is restricted to the declaration ----------------------


def test_a_feature_cannot_read_a_column_it_did_not_declare():
    """The mistake that actually happens: body edited, declaration left alone."""

    @feature("sneaky", "D1", "code")
    def sneaky(view: Mapping[str, object]) -> float:
        return float(view["visible_passed"] or 0)

    with pytest.raises(LeakageError, match="did not declare"):
        sneaky.compute(a_row())


def test_the_message_names_the_feature_and_what_it_declared():
    @feature("sneaky", "D1", "code")
    def sneaky(view: Mapping[str, object]) -> float:
        return float(view["decode_tokens"] or 0)

    with pytest.raises(LeakageError) as excinfo:
        sneaky.compute(a_row())
    message = str(excinfo.value)
    assert "'sneaky'" in message
    assert "decode_tokens" in message
    assert "['code']" in message


def test_an_undeclared_column_raises_rather_than_reading_as_none():
    """Returning None would let the feature emit a plausible constant."""
    view = RowView(a_row(), ("code",), "f")
    assert view["code"].startswith("def solve")
    with pytest.raises(LeakageError):
        view["task_prompt"]


def test_a_declared_column_that_is_absent_reads_as_none():
    """Declared-but-missing is a data question, not a leakage question."""
    view = RowView({"code": "x"}, ("code", "error"), "f")
    assert view["error"] is None


# -- feature sets ------------------------------------------------------------


def test_a_set_spanning_decision_points_is_refused():
    d0 = Feature("a", "D0", ("task_prompt",), lambda v: 1.0)
    d1 = Feature("b", "D1", ("code",), lambda v: 1.0)
    with pytest.raises(FeatureError, match="spans decision points"):
        FeatureSet([d0, d1])


def test_duplicate_feature_names_are_refused():
    one = Feature("same", "D0", ("task_prompt",), lambda v: 1.0)
    two = Feature("same", "D0", ("task_entrypoint",), lambda v: 2.0)
    with pytest.raises(FeatureError, match="duplicate"):
        FeatureSet([one, two])


def test_building_without_the_task_join_says_so(caplog):
    """The prompt is not on the rollout row, and the error must say that."""
    rows = [{k: v for k, v in a_row().items() if not k.startswith("task_")}]
    with pytest.raises(FeatureError, match="rollout row.*carries no prompt"):
        D0_FEATURES.build(rows)


def test_a_non_finite_feature_is_refused():
    @feature("explodes", "D0", "task_prompt")
    def explodes(view: Mapping[str, object]) -> float:
        return float("nan")

    with pytest.raises(FeatureError, match="NaN"):
        FeatureSet([explodes]).build([a_row()])


def test_d1_is_a_strict_superset_of_d0():
    """`learned_D0 -> learned_D1` isolates information, so nothing else may vary."""
    assert set(D0_FEATURES.names) < set(D1_FEATURES.names)


def test_the_probe_set_is_not_on_by_default():
    """Its features oblige the policy to pay for draws it would not otherwise take."""
    assert set(PROBE_FEATURES.names).isdisjoint(D1_FEATURES.names)
    assert feature_set("D1").names == D1_FEATURES.names
    assert set(feature_set("D1", with_probe=True).names) > set(D1_FEATURES.names)


def test_the_probe_cannot_be_requested_at_d0():
    with pytest.raises(FeatureError, match="already happened"):
        feature_set("D0", with_probe=True)


def test_paid_features_declare_the_arm_they_charge_for():
    assert PROBE_FEATURES.paid_arms == ("probe_small",)
    assert D1_FEATURES.paid_arms == ()


def test_a_paid_feature_that_reads_no_siblings_is_refused():
    """Paying for draws you never look at is a declaration bug, not a cost."""
    with pytest.raises(FeatureError, match="costs nothing extra"):
        Feature("mislabelled", "D1", ("code",), lambda v: 1.0,
                paid_arms=("probe_small",))


# -- siblings ----------------------------------------------------------------


def test_a_row_is_never_its_own_sibling():
    """Including it folds the outcome being predicted into the prediction."""

    @feature("count_siblings", "D1", "visible_total", needs_siblings=True)
    def count_siblings(view, siblings: Sequence[Mapping[str, object]]) -> float:
        return float(len(siblings))

    rows = [a_row(rollout_id=f"r{s}", seed=s) for s in range(3)]
    matrix = FeatureSet([count_siblings]).build(rows)
    assert list(matrix.column("count_siblings")) == [2.0, 2.0, 2.0]


def test_identical_sibling_draws_are_not_collapsed():
    """Two seeds that agreed are two observations, not one.

    They are equal dicts, so an equality-based exclusion would drop both and
    quietly shrink the sibling set exactly where the seeds agreed — biasing the
    feature toward disagreement.
    """

    @feature("count_siblings", "D1", "visible_total", needs_siblings=True)
    def count_siblings(view, siblings) -> float:
        return float(len(siblings))

    rows = [a_row(rollout_id="r0"), a_row(rollout_id="r0")]
    matrix = FeatureSet([count_siblings]).build(rows)
    assert list(matrix.column("count_siblings")) == [1.0, 1.0]


def test_siblings_are_grouped_by_task_and_arm():
    @feature("count_siblings", "D1", "visible_total", needs_siblings=True)
    def count_siblings(view, siblings) -> float:
        return float(len(siblings))

    rows = [
        a_row(rollout_id="a0", task_id="t1", arm="small"),
        a_row(rollout_id="a1", task_id="t1", arm="small"),
        a_row(rollout_id="b0", task_id="t1", arm="large"),
        a_row(rollout_id="c0", task_id="t2", arm="small"),
    ]
    matrix = FeatureSet([count_siblings]).build(rows)
    assert list(matrix.column("count_siblings")) == [1.0, 1.0, 0.0, 0.0]


def test_a_sibling_feature_cannot_read_undeclared_columns_either():
    @feature("peeks", "D1", "visible_total", needs_siblings=True)
    def peeks(view, siblings) -> float:
        return float(siblings[0]["code"] != "") if siblings else 0.0

    rows = [a_row(rollout_id="r0"), a_row(rollout_id="r1")]
    with pytest.raises(LeakageError, match="did not declare"):
        FeatureSet([peeks]).build(rows)


# -- the spec file R4 audits -------------------------------------------------


def test_the_spec_records_every_column_the_set_reads(tmp_path):
    import json

    path = D1_FEATURES.write_spec(tmp_path / "feature_spec.json",
                                  run_id="2026-01-01-abc1234-def567")
    spec = json.loads(path.read_text())

    assert spec["decision_point"] == "D1"
    assert spec["run_id"] == "2026-01-01-abc1234-def567"
    assert set(spec["source_columns"]) == set(D1_FEATURES.source_columns)
    assert {f["name"] for f in spec["features"]} == set(D1_FEATURES.names)
    for declared in spec["features"]:
        assert declared["decision_point"] == "D1"


def test_no_declared_column_anywhere_is_a_label():
    """The audit R4 runs independently, run here so it is never news."""
    for name, fs in (("D0", D0_FEATURES), ("D1", D1_FEATURES),
                     ("probe", PROBE_FEATURES)):
        contract.assert_no_labels(fs.source_columns, context=f"{name} feature set")


def test_every_d0_column_is_observable_before_generating():
    assert set(D0_FEATURES.source_columns) <= contract.D0_OBSERVABLE
