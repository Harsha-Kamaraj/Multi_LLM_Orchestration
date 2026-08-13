"""The heuristic must be a real competitor, tuned honestly.

Two failure modes are tested for specifically, because both make the ablation
meaningless while leaving the numbers looking fine: a heuristic that was never
tuned (a strawman), and a heuristic with enough parameters to be a model in
disguise (a duplicate of `learned_D0`).
"""

from __future__ import annotations

import pytest

from eval.heuristics import (
    MAX_FREE_PARAMETERS,
    ThresholdRule,
    candidate_thresholds,
    features_from_tasks,
    prompt_features,
    synth_features,
    tune,
    tuned_frontier,
)
from eval.policies import always, standard_baselines

LAM = 0.05


@pytest.fixture(scope="module")
def features(synth):
    return synth_features(synth.rows)


def test_tuning_finds_the_planted_prompt_signal(store, features):
    """The fixture plants a weak prompt-only proxy. A tuner that cannot beat
    the cheap fixed arm on data containing a real D0 signal has a bug."""
    best = tune(features, store, LAM)
    small = always("small")(store).per_task().utility(LAM).mean()
    assert best.fit_utility > small


def test_tuned_heuristic_beats_random_routing(store, features):
    """Random routing controls for 'any routing helps'. If the heuristic cannot
    beat it, the prompt signal is not being used at all."""
    best = tune(features, store, LAM)
    heuristic = best.rule.policy(features)(store).per_task().utility(LAM).mean()
    rand = standard_baselines()["random_route(0.5)"](store).per_task().utility(LAM).mean()
    assert heuristic > rand


def test_the_rule_stays_a_heuristic(store, features):
    """More free parameters than this and it is a model wearing a disguise —
    which would make the learning ablation compare a model against a model."""
    best = tune(features, store, LAM)
    assert best.rule.n_free_parameters <= MAX_FREE_PARAMETERS


def test_tuning_records_which_split_it_saw(store, features):
    """A report must never claim a held-out number for a rule tuned on the
    split it is reported against."""
    best = tune(features, store, LAM)
    assert best.fit_split == store.splits


def test_tuning_at_high_lambda_prefers_the_cheap_arm(store, features):
    """When cost dominates, the tuned rule should escalate rarely. A rule that
    ignores λ is not being tuned per-λ, and the frontier would be a dot."""
    best = tune(features, store, lam=1.0)
    actions = best.rule.actions(features, store.ordered_tasks)
    share_large = sum(a == "large" for a in actions.values()) / len(actions)
    assert share_large < 0.15


def test_tuning_at_zero_lambda_prefers_accuracy(store, features):
    best = tune(features, store, lam=0.0)
    actions = best.rule.actions(features, store.ordered_tasks)
    share_large = sum(a == "large" for a in actions.values()) / len(actions)
    assert share_large > 0.85


def test_frontier_produces_a_curve_not_a_point(store, features):
    """A single tuned threshold compared against a λ-swept policy is a dot
    against a curve, and the curve wins for free."""
    lams = [0.0, 0.02, 0.05, 0.1, 0.3]
    frontier = tuned_frontier(features, store, store, lams)
    assert len(frontier) == len(lams)
    escalation = [
        sum(1 for a in t.rule.actions(features, store.ordered_tasks).values()
            if a == "large")
        for _, t, _ in frontier
    ]
    assert escalation == sorted(escalation, reverse=True), (
        "escalation share must fall as cost weight rises"
    )


def test_tuning_on_one_split_and_scoring_on_another_is_supported(synth, features):
    """The honest configuration: fit on val, report on a different split."""
    from eval import load_rows
    from eval.loading import TestSplitUnlock

    fit = load_rows(synth.rows, splits=("val",))
    held = load_rows(
        synth.rows, splits=("test",),
        unlock=TestSplitUnlock(reason="suite", preregistration=__file__),
    )
    best = tune(features, fit, LAM)
    outcome = best.rule.policy(features)(held)
    assert outcome.n_tasks == held.n_tasks
    assert best.fit_split == ("val",)


def test_missing_feature_is_an_error_not_a_default(store, features):
    """Defaulting would attribute the default's outcome to the heuristic."""
    partial = {k: v for k, v in list(features.items())[:-5]}
    rule = ThresholdRule("difficulty_proxy", 0.0, True)
    with pytest.raises(KeyError, match="must decide for every task"):
        rule.actions(partial, store.ordered_tasks)


def test_candidate_thresholds_are_quantile_spaced():
    """A uniform grid over a skewed feature wastes candidates in an empty tail,
    and the heuristic would lose for a reason unrelated to learning."""
    import numpy as np

    skewed = np.concatenate([np.zeros(900), np.linspace(1, 1000, 100)])
    cuts = candidate_thresholds(skewed, n=20)
    assert (cuts <= 1.0).sum() <= len(cuts) // 2 + 2
    assert len(cuts) >= 2


def test_candidate_thresholds_handle_an_empty_feature():
    import numpy as np

    assert candidate_thresholds(np.array([np.nan, np.nan])).size == 0


# --- real prompt features --------------------------------------------------

def test_prompt_features_are_prompt_only():
    """No feature may read hidden tests or a pass-rate-derived difficulty."""
    task = {
        "task_id": "t/1",
        "prompt": "def f(x):\n    '''Return the shortest path in a graph.'''\n",
        "visible_tests": "assert f(1) == 1\nassert f(2) == 2\n",
        "hidden_tests": "assert f(999) == 999\n" * 50,
        "difficulty": 0.99,
    }
    feats = prompt_features(task)
    assert feats["n_visible_tests"] == 2.0
    assert feats["algorithmic_terms"] >= 1.0
    assert all(v < 100 for k, v in feats.items() if k != "prompt_chars")
    # The hidden suite is huge; nothing may reflect its size.
    assert "hidden" not in " ".join(feats)
    assert "difficulty" not in " ".join(feats)


def test_features_from_tasks_keys_by_task_id():
    tasks = [
        {"task_id": "a", "prompt": "x", "visible_tests": ""},
        {"task_id": "b", "prompt": "y y y", "visible_tests": "assert 1"},
    ]
    table = features_from_tasks(tasks)
    assert set(table) == {"a", "b"}
    assert table["b"]["prompt_words"] == 3.0
