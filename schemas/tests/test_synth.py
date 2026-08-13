"""The fixture generator must plant the signal it claims to plant.

If these fail, every test that uses the generator is asserting against the
wrong ground truth — and it will still pass, which is the problem.
"""

from __future__ import annotations

import numpy as np

from schemas import SynthConfig, generate
from schemas.synth import _auc


def test_generation_is_deterministic():
    a = generate(SynthConfig(n_tasks=50), seed=3).rows
    b = generate(SynthConfig(n_tasks=50), seed=3).rows
    assert a == b


def test_different_seeds_differ():
    a = generate(SynthConfig(n_tasks=50), seed=3).rows
    b = generate(SynthConfig(n_tasks=50), seed=4).rows
    assert a != b


def test_config_change_changes_run_id():
    """Two fixture sets from different settings must never collide in a store —
    the same discipline the real sweep uses."""
    a = SynthConfig(n_tasks=50, d0_signal=0.5)
    b = SynthConfig(n_tasks=50, d0_signal=0.9)
    assert a.run_id != b.run_id


def test_large_arm_beats_small_by_the_planted_gap(synth):
    solved = synth.truth["solved"]
    gap = solved["large"].mean() - solved["small"].mean()
    # The Phase 0 gate is 8pp; the fixture plants roughly the real 1.5B/7B gap.
    assert gap > 0.08, f"planted arm gap {gap:.3f} is below the Phase 0 gate"


def test_d0_signal_is_weak_as_intended(synth):
    """D0 must be weak in the fixture. A generator that flattered pre-generation
    routing would hide the asymmetry the whole project reports."""
    auc = synth.planted_auc_d0()
    assert 0.58 < auc < 0.75, f"planted AUC_D0 {auc:.3f} outside the honest band"


def test_d1_signal_is_strong(synth):
    """Visible-test outcome must predict the hidden outcome far better than the
    prompt does — observing beats predicting."""
    small = [r for r in synth.rows if r["arm"] == "small"]
    visible = np.array([r["visible_passed"] / r["visible_total"] for r in small])
    hidden = np.array([r["hidden_passed"] == r["hidden_total"] for r in small])
    auc = _auc(visible, hidden)
    assert auc > 0.78, f"planted AUC_D1 {auc:.3f} too weak to be the D1 signal"
    assert auc > synth.planted_auc_d0(), "D1 must dominate D0 by construction"


def test_splits_are_task_level(synth):
    """Every seed of one task lands in the same split. Row-level splitting puts
    near-copies of training rows in the test set and narrows every interval."""
    by_task: dict[str, set[str]] = {}
    for row in synth.rows:
        by_task.setdefault(row["task_id"], set()).add(row["split"])
    assert all(len(s) == 1 for s in by_task.values())


def test_split_proportions_are_roughly_60_20_20(synth):
    tasks = {r["task_id"]: r["split"] for r in synth.rows}
    counts = {s: sum(1 for v in tasks.values() if v == s) for s in ("train", "val", "test")}
    n = len(tasks)
    assert abs(counts["train"] / n - 0.6) < 0.02
    assert abs(counts["val"] / n - 0.2) < 0.02
    assert abs(counts["test"] / n - 0.2) < 0.02


def test_every_task_has_every_arm_and_seed(synth):
    """Paired comparisons assume a complete grid. A hole in it silently turns a
    paired test into an unpaired one."""
    cfg = synth.truth["config"]
    cells: dict[str, set[tuple[str, int]]] = {}
    for row in synth.rows:
        cells.setdefault(row["task_id"], set()).add((row["arm"], row["seed"]))
    expected = {(a.name, s) for a in cfg.arms for s in range(cfg.seeds)}
    assert all(c == expected for c in cells.values())


def test_rollout_ids_are_unique(synth):
    ids = [r["rollout_id"] for r in synth.rows]
    assert len(ids) == len(set(ids))


def test_optimal_policy_prefers_small_when_cost_dominates(synth):
    """At a large λ the optimal router should almost never pay for the large arm.
    This is the sanity check on the ground truth itself."""
    actions = synth.optimal_actions(lam=1.0)
    share_large = sum(a == "large" for a in actions.values()) / len(actions)
    assert share_large < 0.05


def test_optimal_policy_prefers_large_when_cost_is_free(synth):
    actions = synth.optimal_actions(lam=0.0)
    share_large = sum(a == "large" for a in actions.values()) / len(actions)
    assert share_large > 0.95


def test_optimal_value_is_monotone_decreasing_in_lambda(synth):
    values = [synth.optimal_value(lam) for lam in (0.0, 0.02, 0.05, 0.1, 0.3)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_optimal_policy_beats_both_fixed_arms(synth):
    """The planted ceiling must actually be a ceiling. If a fixed arm matched
    it, there would be no routing problem to solve and every downstream test
    comparing against 'optimal' would be vacuous."""
    lam = 0.05
    d = synth.truth["difficulty"]
    specs = {a.name: a for a in synth.truth["arms"]}
    fixed = {
        name: float((spec.p_solve(d) - lam * spec.cost).mean())
        for name, spec in specs.items()
    }
    assert synth.optimal_value(lam) > max(fixed.values()) + 1e-9


def test_wall_ms_is_not_usable_as_latency(synth):
    """The fixture makes sweep wall-clock visibly unlike imputed latency on
    purpose, so a consumer that confuses them fails loudly here rather than
    quietly in a report."""
    wall = np.array([r["wall_ms"] for r in synth.rows]) / 1000.0
    imputed = np.array([r["imputed_latency_s"] for r in synth.rows])
    assert abs(np.corrcoef(wall, imputed)[0, 1]) < 0.2
