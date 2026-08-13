"""The D0 and D1 sets, built on the fixture and checked one feature at a time.

Two kinds of test here. The unit tests pin individual features against
hand-made rows, where the right answer is obvious by inspection. The last
section builds both sets on the synthetic store and checks the property the
whole project rests on: that D1 carries far more signal than D0.

That asymmetry is asserted as a *direction*, not a magnitude. Pinning
`AUC_D1 > 0.80` here would be pinning R4's fixture parameters, and a test that
fails when someone tunes `d1_fidelity` is a test about the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orchestrator.policy import fixtures, store
from orchestrator.policy.features import FeatureSet
from orchestrator.policy.features.d0 import D0_FEATURES
from orchestrator.policy.features.d1 import D1_FEATURES, PROBE_FEATURES
from schemas.synth import SynthConfig

from .test_features_spec import a_row


@pytest.fixture(scope="module")
def loaded(tmp_path_factory) -> store.RolloutData:
    root = tmp_path_factory.mktemp("store")
    fx = fixtures.write_fixture(root, SynthConfig(n_tasks=250, seeds=3))
    return store.load_rollouts(
        fx.root, fx.run_id,
        tasks_path=fx.tasks_path,
        cost_fingerprint=fx.cost_fingerprint,
    )


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores = np.asarray(scores)[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def one(feature) -> FeatureSet:
    return FeatureSet([feature])


# -- D0, one feature at a time -----------------------------------------------


def test_prompt_words_counts_words():
    from orchestrator.policy.features.d0 import prompt_words
    assert prompt_words.compute(a_row(task_prompt="one two three")) == 3.0


def test_prompt_sentences_ignores_trailing_punctuation():
    from orchestrator.policy.features.d0 import prompt_sentences
    assert prompt_sentences.compute(a_row(task_prompt="One. Two! Three?")) == 3.0
    assert prompt_sentences.compute(a_row(task_prompt="")) == 0.0


def test_n_visible_tests_counts_non_blank_lines():
    from orchestrator.policy.features.d0 import n_visible_tests
    row = a_row(task_visible_tests="assert a\n\nassert b\n")
    assert n_visible_tests.compute(row) == 2.0


def test_algorithmic_keywords_finds_the_listed_words():
    from orchestrator.policy.features.d0 import algorithmic_keywords
    row = a_row(task_prompt="Find the shortest path in the graph.")
    assert algorithmic_keywords.compute(row) == 2.0


def test_digit_density_is_a_fraction():
    from orchestrator.policy.features.d0 import digit_density
    assert digit_density.compute(a_row(task_prompt="ab12")) == 0.5
    assert digit_density.compute(a_row(task_prompt="")) == 0.0


def test_entrypoint_words_splits_on_underscores():
    from orchestrator.policy.features.d0 import entrypoint_words
    assert entrypoint_words.compute(a_row(task_entrypoint="find_max_sum")) == 3.0


# -- D1, one feature at a time -----------------------------------------------


def test_visible_pass_rate_is_the_fraction():
    from orchestrator.policy.features.d1 import visible_pass_rate
    assert visible_pass_rate.compute(a_row(visible_passed=3, visible_total=4)) == 0.75


def test_no_visible_suite_is_distinguishable_from_failing_all_of_it():
    """Zero-of-zero and zero-of-four are different states, and the pair says so."""
    from orchestrator.policy.features.d1 import has_visible_tests, visible_pass_rate

    none = a_row(visible_passed=None, visible_total=0)
    failed = a_row(visible_passed=0, visible_total=4)

    assert visible_pass_rate.compute(none) == visible_pass_rate.compute(failed) == 0.0
    assert has_visible_tests.compute(none) == 0.0
    assert has_visible_tests.compute(failed) == 1.0


def test_visible_all_passed_is_what_the_cascade_gates_on():
    from orchestrator.policy.features.d1 import visible_all_passed
    assert visible_all_passed.compute(a_row(visible_passed=4, visible_total=4)) == 1.0
    assert visible_all_passed.compute(a_row(visible_passed=3, visible_total=4)) == 0.0
    assert visible_all_passed.compute(a_row(visible_passed=0, visible_total=0)) == 0.0


def test_code_ast_nodes_counts_structure():
    from orchestrator.policy.features.d1 import code_ast_nodes
    assert code_ast_nodes.compute(a_row(code="x = 1\n")) > 0
    assert code_ast_nodes.compute(a_row(code="")) == 0.0


def test_unparseable_code_scores_zero_structure_rather_than_raising():
    from orchestrator.policy.features.d1 import code_ast_nodes, code_max_depth
    broken = a_row(code="def f(:\n")
    assert code_ast_nodes.compute(broken) == 0.0
    assert code_max_depth.compute(broken) == 0.0


def test_defines_entrypoint_reads_the_ast():
    from orchestrator.policy.features.d1 import defines_entrypoint
    assert defines_entrypoint.compute(
        a_row(code="def solve(x):\n    return x\n", task_entrypoint="solve")) == 1.0
    assert defines_entrypoint.compute(
        a_row(code="def other(x):\n    return x\n", task_entrypoint="solve")) == 0.0


def test_defines_entrypoint_falls_back_to_text_when_code_is_broken():
    """Broken-but-defines-it and never-defines-it are different escalations."""
    from orchestrator.policy.features.d1 import defines_entrypoint
    row = a_row(code="def solve(x:\n    return x\n", task_entrypoint="solve")
    assert defines_entrypoint.compute(row) == 1.0


def test_finish_reason_indicators_are_mutually_exclusive():
    from orchestrator.policy.features.d1 import FINISH_FEATURES
    row = a_row(finish_reason="length")
    values = [f.compute(row) for f in FINISH_FEATURES]
    assert sum(values) == 1.0
    assert dict(zip([f.name for f in FINISH_FEATURES], values))["finish_length"] == 1.0


def test_hack_flags_count_survives_an_empty_tuple():
    from orchestrator.policy.features.d1 import hack_flag_count
    assert hack_flag_count.compute(a_row(hack_flags=())) == 0.0
    assert hack_flag_count.compute(a_row(hack_flags=("hardcoded", "skip"))) == 2.0


def test_sibling_agreement_is_neutral_with_no_siblings():
    """0.5 is honest. A 0 or 1 default is a confident claim from no evidence."""
    from orchestrator.policy.features.d1 import sibling_visible_agreement
    assert sibling_visible_agreement.compute(a_row(), ()) == 0.5


# -- built on the fixture ----------------------------------------------------


def test_both_sets_build_on_the_fixture(loaded):
    d0 = D0_FEATURES.build(loaded.rows)
    d1 = D1_FEATURES.build(loaded.rows)
    assert d0.shape == (len(loaded), len(D0_FEATURES))
    assert d1.shape == (len(loaded), len(D1_FEATURES))
    assert d0.rollout_ids == d1.rollout_ids


def test_the_matrix_stays_keyed_to_the_rows_it_came_from(loaded):
    """Phase 5 joins these to labels by rollout_id, so the order must hold."""
    matrix = D1_FEATURES.build(loaded.rows)
    assert matrix.rollout_ids == tuple(r["rollout_id"] for r in loaded.rows)


def test_the_probe_adds_columns_without_disturbing_the_others(loaded):
    plain = D1_FEATURES.build(loaded.rows)
    probed = (D1_FEATURES + PROBE_FEATURES).build(loaded.rows)
    assert probed.shape[1] == plain.shape[1] + len(PROBE_FEATURES)
    np.testing.assert_array_equal(
        probed.X[:, :plain.shape[1]], plain.X
    )


def test_the_fixture_cannot_exercise_the_code_shape_features(loaded):
    """A known limitation, asserted so it stays known.

    `schemas.synth` emits one templated code string per row, so every feature
    that measures the *candidate* is constant on the fixture. They are real
    features that will vary on a real store, and nothing here validates them
    beyond their unit tests above. Worth stating: a green suite on synthetic
    data is not evidence that the D1 code features carry signal.
    """
    matrix = D1_FEATURES.build(loaded.rows)
    constant = set(matrix.constant_columns())
    assert {"code_chars", "code_lines", "code_ast_nodes"} <= constant
    # The visible-test features are the ones the fixture does plant signal in.
    assert "visible_pass_rate" not in constant


def test_a_prompt_feature_recovers_the_planted_d0_proxy(loaded):
    """No fixture-specific column involved — just prompt length."""
    from orchestrator.policy.features.d0 import prompt_words

    values = one(prompt_words).build(loaded.rows).column("prompt_words")
    proxy = np.array([row["task_x_d0"] for row in loaded.rows])
    # Negative: longer prompt, harder task, lower proxy. See `fixtures._prompt_for`.
    assert np.corrcoef(values, proxy)[0, 1] < -0.99


# -- the asymmetry that drives the architecture ------------------------------


def test_d1_carries_more_signal_than_d0(loaded):
    """The most important number in the project, as a direction.

    Observing failure beats predicting it. Asserted as an ordering rather than
    a threshold: pinning a magnitude here would pin R4's fixture parameters,
    and this test would then fail whenever someone tuned `d1_fidelity` — which
    is a knob, not a regression.

    The Phase 0 gate numbers themselves are measured in Phase 4, with intervals,
    from a fitted model rather than from single features.
    """
    from orchestrator.policy.features.d0 import prompt_words
    from orchestrator.policy.features.d1 import visible_pass_rate

    small = [row for row in loaded.rows if row["arm"] == "small"]
    solved = np.array([loaded.label_for(r["rollout_id"]).solved for r in small])

    # Prompt length runs the other way — longer prompt, harder task — so the
    # D0 score is negated. An inverted feature is a real bug that shows up as
    # AUC below 0.5, and this is the one place it is expected rather than a
    # symptom.
    d0_auc = auc(-one(prompt_words).build(small).column("prompt_words"), solved)
    d1_auc = auc(one(visible_pass_rate).build(small).column("visible_pass_rate"),
                 solved)

    assert d0_auc > 0.55, f"D0 proxy carries no signal at all: {d0_auc:.3f}"
    assert d1_auc > d0_auc + 0.10, (
        f"D1 ({d1_auc:.3f}) should dominate D0 ({d0_auc:.3f}) by a wide margin. "
        f"If it does not, either the fixture's d1_fidelity was lowered or the "
        f"visible-test features stopped reading the outcome."
    )
