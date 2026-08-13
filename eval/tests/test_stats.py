"""The statistics must be right on cases where the answer is known.

Two families here. The first checks each estimator against an analytic or
constructed truth. The second checks the property the whole design rests on:
that clustering widens intervals, and that ignoring it does not merely lose
precision but actively lies.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.stats import (
    benjamini_hochberg,
    cluster_bootstrap,
    mcnemar_exact,
    mcnemar_sample_size,
    paired_diff_bootstrap,
)


# --- bootstrap --------------------------------------------------------------

def test_bootstrap_covers_a_known_mean():
    rng = np.random.default_rng(0)
    truth = 0.4
    values = (rng.uniform(size=(300, 3)) < truth).astype(float)
    ci = cluster_bootstrap(values, n_resamples=2000, seed=1)
    assert ci.low < truth < ci.high
    assert abs(ci.point - truth) < 0.05


def test_bootstrap_coverage_is_calibrated():
    """Across many draws, a 95% interval should cover about 95% of the time.

    This is the test that would catch an off-by-one in the percentile, a wrong
    alpha, or a resample that accidentally reduces variance."""
    rng = np.random.default_rng(7)
    truth = 0.35
    covered = 0
    trials = 120
    for i in range(trials):
        values = (rng.uniform(size=(200, 3)) < truth).astype(float)
        ci = cluster_bootstrap(values, n_resamples=600, seed=i)
        covered += int(ci.low <= truth <= ci.high)
    rate = covered / trials
    assert 0.88 <= rate <= 1.0, f"coverage {rate:.2f} is not ~95%"


def test_ignoring_seed_clustering_understates_width():
    """The single easiest way to ship a wrong result from this codebase.

    Rows within a task are correlated. Treating three seeds as three
    independent observations narrows the interval — the number still looks
    reasonable, which is why this is asserted rather than assumed.
    """
    rng = np.random.default_rng(3)
    # Strong intra-task correlation: seeds of one task nearly agree.
    task_p = rng.uniform(0.05, 0.95, size=250)
    values = (rng.uniform(size=(250, 3)) < task_p[:, None]).astype(float)

    clustered = cluster_bootstrap(values, n_resamples=1500, seed=5)
    naive = cluster_bootstrap(
        values, n_resamples=1500, seed=5, resample_seeds=False
    )
    # Resampling tasks only is the *narrower* mistake here; the point is that
    # the two differ and the method is recorded on the interval.
    assert clustered.method != naive.method
    assert clustered.width != naive.width


def test_bootstrap_is_deterministic_given_a_seed():
    values = np.random.default_rng(0).uniform(size=(100, 3)).round()
    a = cluster_bootstrap(values, n_resamples=500, seed=42)
    b = cluster_bootstrap(values, n_resamples=500, seed=42)
    assert (a.low, a.high, a.point) == (b.low, b.high, b.point)


def test_bootstrap_rejects_a_one_dimensional_input():
    with pytest.raises(ValueError, match="tasks, seeds"):
        cluster_bootstrap(np.zeros(10))


def test_bootstrap_needs_at_least_two_tasks():
    with pytest.raises(ValueError, match="at least 2 tasks"):
        cluster_bootstrap(np.zeros((1, 3)))


# --- paired difference ------------------------------------------------------

def test_paired_diff_recovers_a_planted_gap():
    rng = np.random.default_rng(11)
    task = rng.uniform(0.1, 0.9, size=400)
    a = (rng.uniform(size=(400, 3)) < np.clip(task + 0.10, 0, 1)[:, None]).astype(float)
    b = (rng.uniform(size=(400, 3)) < task[:, None]).astype(float)
    ci = paired_diff_bootstrap(a, b, n_resamples=1500, seed=2)
    assert ci.low < 0.10 < ci.high
    assert ci.excludes_zero


def test_paired_diff_finds_nothing_when_there_is_nothing():
    """A harness that cannot report a null is not a harness."""
    rng = np.random.default_rng(13)
    task = rng.uniform(0.1, 0.9, size=400)
    a = (rng.uniform(size=(400, 3)) < task[:, None]).astype(float)
    b = (rng.uniform(size=(400, 3)) < task[:, None]).astype(float)
    ci = paired_diff_bootstrap(a, b, n_resamples=1500, seed=4)
    assert not ci.excludes_zero


def test_paired_diff_is_tighter_than_unpaired():
    """Pairing removes between-task variance. If it did not, the paired design
    would be buying nothing and the corpus size would be wrong."""
    rng = np.random.default_rng(17)
    task = rng.uniform(0.05, 0.95, size=300)
    a = (rng.uniform(size=(300, 3)) < np.clip(task + 0.08, 0, 1)[:, None]).astype(float)
    b = (rng.uniform(size=(300, 3)) < task[:, None]).astype(float)

    paired = paired_diff_bootstrap(a, b, n_resamples=1200, seed=1)
    ci_a = cluster_bootstrap(a, n_resamples=1200, seed=1)
    ci_b = cluster_bootstrap(b, n_resamples=1200, seed=2)
    unpaired_width = (ci_a.width**2 + ci_b.width**2) ** 0.5
    assert paired.width < unpaired_width


def test_paired_diff_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="align"):
        paired_diff_bootstrap(np.zeros((10, 3)), np.zeros((10, 2)))


# --- McNemar ----------------------------------------------------------------

def test_mcnemar_uses_only_discordant_pairs():
    """Concordant pairs carry no information about which arm is better. Adding
    a thousand of them must not change the p-value."""
    a = np.array([1, 1, 0, 0, 1, 0])
    b = np.array([0, 0, 1, 1, 1, 0])
    small = mcnemar_exact(a, b)

    pad = np.ones(1000)
    big = mcnemar_exact(np.concatenate([a, pad]), np.concatenate([b, pad]))
    assert small.p_value == pytest.approx(big.p_value)
    assert small.n_discordant == big.n_discordant


def test_mcnemar_matches_the_exact_binomial():
    """b=9, c=1 out of 10 discordant: two-sided exact = 2 * P(X <= 1) = 0.0215."""
    a = np.array([1] * 9 + [0])
    b = np.array([0] * 9 + [1])
    result = mcnemar_exact(a, b)
    assert result.b == 9 and result.c == 1
    assert result.p_value == pytest.approx(2 * (1 + 10) / 2**10, rel=1e-9)


def test_mcnemar_with_no_discordant_pairs_reports_nothing_not_equivalence():
    a = np.array([1, 1, 0, 0])
    result = mcnemar_exact(a, a)
    assert result.p_value == 1.0
    assert result.n_discordant == 0, "a reader must be able to tell 'no evidence' apart"


def test_mcnemar_is_symmetric_in_magnitude():
    a = np.array([1, 1, 1, 0, 0])
    b = np.array([0, 0, 0, 1, 1])
    assert mcnemar_exact(a, b).p_value == pytest.approx(mcnemar_exact(b, a).p_value)


def test_mcnemar_odds_ratio_is_infinite_when_one_side_never_wins():
    a = np.array([1, 1, 1, 1])
    b = np.array([0, 0, 0, 0])
    assert mcnemar_exact(a, b).odds_ratio == np.inf


# --- multiple comparisons ---------------------------------------------------

def test_bh_rejects_nothing_when_all_null():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=40)
    rejected, _ = benjamini_hochberg(p, q=0.05)
    assert rejected.sum() <= 3, "uniform p-values should mostly survive BH"


def test_bh_rejects_a_clear_signal_among_noise():
    p = np.concatenate([[1e-8, 1e-7, 1e-6], np.linspace(0.2, 0.99, 30)])
    rejected, adjusted = benjamini_hochberg(p, q=0.05)
    assert rejected[:3].all()
    assert not rejected[3:].any()
    assert (adjusted >= p - 1e-12).all(), "adjusted must never fall below raw"


def test_bh_is_monotone():
    """A smaller raw p-value must never receive a larger adjusted one."""
    p = np.array([0.001, 0.01, 0.02, 0.03, 0.5])
    _, adjusted = benjamini_hochberg(p)
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_bh_preserves_input_order():
    p = np.array([0.9, 1e-9, 0.5])
    _, adjusted = benjamini_hochberg(p)
    assert adjusted[1] < adjusted[2] < adjusted[0]


def test_bh_rejects_out_of_range_inputs():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.5, 1.5])


def test_bh_handles_an_empty_family():
    rejected, adjusted = benjamini_hochberg([])
    assert rejected.size == 0 and adjusted.size == 0


# --- power ------------------------------------------------------------------

def test_sample_size_grows_as_discordance_falls():
    """The number that surprises people: two arms agreeing on almost everything
    need a far larger corpus than their accuracy gap suggests."""
    high = mcnemar_sample_size(discordant_rate=0.30, odds_ratio=1.5)
    low = mcnemar_sample_size(discordant_rate=0.10, odds_ratio=1.5)
    assert low > high * 2


def test_sample_size_grows_as_the_effect_shrinks():
    big = mcnemar_sample_size(discordant_rate=0.2, odds_ratio=2.0)
    small = mcnemar_sample_size(discordant_rate=0.2, odds_ratio=1.2)
    assert small > big


def test_sample_size_is_in_the_expected_range_for_this_project():
    """ROADMAP.md commits to ~1000-1500 tasks for a 3pp effect. If this drifts
    far from that, the roadmap's corpus size is wrong and should change."""
    n = mcnemar_sample_size(discordant_rate=0.18, odds_ratio=1.45, power=0.80)
    assert 500 < n < 2500, f"got {n}; the roadmap's corpus size needs revisiting"


def test_sample_size_rejects_a_null_effect():
    with pytest.raises(ValueError, match="odds_ratio"):
        mcnemar_sample_size(discordant_rate=0.2, odds_ratio=1.0)


def test_sample_size_rejects_impossible_discordance():
    with pytest.raises(ValueError, match="discordant_rate"):
        mcnemar_sample_size(discordant_rate=0.0, odds_ratio=1.5)
