"""Statistics for paired, clustered, multiply-compared rollout data.

Three properties of this data drive every choice here, and getting any of them
wrong produces a confidently wrong result rather than an error:

**It is paired.** Every arm ran on every task. Comparing two independent means
throws that away and inflates the variance enormously — the between-task
variation swamps the between-arm difference we are trying to measure.

**It is clustered.** Three seeds of one task are not three independent
observations. A naive bootstrap over rows treats them as such and returns
intervals that are far too narrow. This is the single easiest way to ship a
wrong result from this codebase, so `cluster_bootstrap` resamples *tasks*, and
seeds within the resampled tasks.

**It is multiply compared.** A λ sweep tests many points on one frontier.
Uncorrected, something will look significant. `benjamini_hochberg` controls the
false discovery rate across the family.

No scipy dependency: normal quantiles come from `statistics.NormalDist`, and
the exact binomial tail is computed directly. R4 needs no GPU and no heavy
install, and that is worth preserving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np

_NORM = NormalDist()


@dataclass(frozen=True)
class Interval:
    """A point estimate with an interval. The only shape a number leaves in."""

    point: float
    low: float
    high: float
    level: float = 0.95
    n_resamples: int = 0
    method: str = "cluster_bootstrap_percentile"

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval supports a directional claim."""
        return (self.low > 0.0) or (self.high < 0.0)

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "point": self.point, "low": self.low, "high": self.high,
            "level": self.level, "n_resamples": self.n_resamples,
            "method": self.method, "excludes_zero": self.excludes_zero,
        }


def cluster_bootstrap(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = lambda a: float(np.nanmean(a)),
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
    resample_seeds: bool = True,
) -> Interval:
    """Bootstrap a statistic over a `(n_tasks, n_seeds)` matrix.

    Tasks are the cluster. Each resample draws `n_tasks` tasks *with
    replacement*, then — because seeds within a task are themselves a random
    sample — draws `n_seeds` seeds with replacement inside each chosen task.

    Resampling only tasks understates variance slightly; resampling only rows
    understates it badly. Both levels are resampled here, which is the design
    the report claims and the reason `resample_seeds` exists as a flag rather
    than an assumption: the test suite compares the two and asserts the naive
    version is visibly narrower.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a (tasks, seeds) matrix, got shape {values.shape}")
    n_tasks, n_seeds = values.shape
    if n_tasks < 2:
        raise ValueError("cluster bootstrap needs at least 2 tasks")

    rng = np.random.default_rng(seed)
    point = statistic(values)

    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        tasks = rng.integers(0, n_tasks, size=n_tasks)
        sample = values[tasks]
        if resample_seeds and n_seeds > 1:
            cols = rng.integers(0, n_seeds, size=(n_tasks, n_seeds))
            sample = np.take_along_axis(sample, cols, axis=1)
        draws[i] = statistic(sample)

    alpha = (1.0 - level) / 2.0
    low, high = np.nanpercentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return Interval(
        point=float(point), low=float(low), high=float(high),
        level=level, n_resamples=n_resamples,
        method="cluster_bootstrap_percentile" if resample_seeds
        else "cluster_bootstrap_tasks_only",
    )


def paired_diff_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Interval for `mean(a) - mean(b)`, resampling the *same* tasks for both.

    Pairing is preserved inside the resample: a task drawn for `a` is the same
    task drawn for `b`. Bootstrapping the two arms independently and
    subtracting would discard the pairing and widen the interval by the
    between-task variance — the exact quantity a paired design exists to
    remove.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")

    n_tasks, n_seeds = a.shape
    rng = np.random.default_rng(seed)
    point = float(np.nanmean(a) - np.nanmean(b))

    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        tasks = rng.integers(0, n_tasks, size=n_tasks)
        sa, sb = a[tasks], b[tasks]
        if n_seeds > 1:
            cols = rng.integers(0, n_seeds, size=(n_tasks, n_seeds))
            sa = np.take_along_axis(sa, cols, axis=1)
            sb = np.take_along_axis(sb, cols, axis=1)
        draws[i] = np.nanmean(sa) - np.nanmean(sb)

    alpha = (1.0 - level) / 2.0
    low, high = np.nanpercentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return Interval(
        point=point, low=float(low), high=float(high),
        level=level, n_resamples=n_resamples, method="paired_cluster_bootstrap",
    )


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar for paired binary outcomes."""

    b: int  # a solved, b did not
    c: int  # b solved, a did not
    p_value: float
    n_discordant: int

    @property
    def odds_ratio(self) -> float:
        """b/c. Infinite when c == 0 and b > 0 — a real, reportable outcome."""
        if self.c == 0:
            return math.inf if self.b > 0 else float("nan")
        return self.b / self.c


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> McNemarResult:
    """Two-sided exact McNemar test on paired binary vectors.

    Only *discordant* pairs carry information: tasks both arms solved, or both
    failed, say nothing about which is better. The exact binomial is used rather
    than the chi-square approximation because the discordant count is often
    small, and the approximation is unreliable exactly there.
    """
    a = np.asarray(a).astype(bool).ravel()
    b = np.asarray(b).astype(bool).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired vectors must align: {a.shape} vs {b.shape}")

    n_b = int(np.sum(a & ~b))
    n_c = int(np.sum(~a & b))
    n = n_b + n_c
    if n == 0:
        # No discordant pairs. Not evidence of equivalence — evidence of
        # nothing, and reported as p=1.0 with n_discordant=0 so a reader can
        # tell the difference.
        return McNemarResult(b=0, c=0, p_value=1.0, n_discordant=0)

    k = min(n_b, n_c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return McNemarResult(b=n_b, c=n_c, p_value=min(1.0, 2.0 * tail), n_discordant=n)


def benjamini_hochberg(
    p_values: Sequence[float], *, q: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Control the false discovery rate at `q`.

    Returns `(rejected, adjusted)`, both in the input order. Adjusted values
    are the standard step-up BH q-values, enforced monotone so a smaller raw
    p-value never yields a larger adjusted one.

    Used across the λ sweep. Testing thirty points on a frontier and reporting
    the best uncorrected is how a null result becomes a headline.
    """
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted_sorted = np.minimum.accumulate(
        (ranked * m / np.arange(1, m + 1))[::-1]
    )[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted <= q, adjusted


def mcnemar_sample_size(
    *,
    discordant_rate: float,
    odds_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Tasks needed to detect `odds_ratio` at `power`, given discordance.

    Connor (1987), the standard closed form for paired binary designs. Note
    what it depends on: the **discordant rate**, not the overall accuracy. Two
    arms that agree on almost everything need a far larger corpus than their
    accuracy gap suggests, which is why this must be computed from a measured
    pilot rather than guessed.

    Do this in week 1. Committing to a corpus size before knowing the
    discordance is committing to an unknown amount of power.
    """
    if not 0 < discordant_rate <= 1:
        raise ValueError("discordant_rate must lie in (0, 1]")
    if odds_ratio <= 0 or odds_ratio == 1:
        raise ValueError("odds_ratio must be positive and != 1")

    z_a = _NORM.inv_cdf(1 - alpha / 2)
    z_b = _NORM.inv_cdf(power)
    psi = odds_ratio
    numerator = (
        z_a * math.sqrt(psi + 1)
        + z_b * math.sqrt((psi + 1) - (psi - 1) ** 2 * discordant_rate)
    ) ** 2
    denominator = (psi - 1) ** 2 * discordant_rate
    return int(math.ceil(numerator / denominator))


def bootstrap_power(
    effect: float,
    *,
    n_tasks: int,
    n_seeds: int = 3,
    base_rate: float = 0.5,
    intra_task_corr: float = 0.6,
    n_trials: int = 400,
    seed: int = 0,
) -> float:
    """Simulated power for the paired bootstrap, under clustering.

    The closed form above ignores seed clustering. This simulates it: outcomes
    within a task are correlated at `intra_task_corr`, which shrinks the
    effective sample size below `n_tasks * n_seeds`. Use it as the sanity check
    on the analytic number — if they disagree badly, the clustering assumption
    is doing more work than expected and should be stated in the report.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_trials):
        task_effect = rng.normal(0, math.sqrt(intra_task_corr), size=n_tasks)
        noise_a = rng.normal(0, math.sqrt(1 - intra_task_corr), (n_tasks, n_seeds))
        noise_b = rng.normal(0, math.sqrt(1 - intra_task_corr), (n_tasks, n_seeds))
        latent_a = task_effect[:, None] + noise_a + effect / max(base_rate, 1e-9) * 0.5
        latent_b = task_effect[:, None] + noise_b
        a = (latent_a > 0).astype(float)
        b = (latent_b > 0).astype(float)
        ci = paired_diff_bootstrap(a, b, n_resamples=300, seed=int(rng.integers(1 << 30)))
        hits += int(ci.excludes_zero)
    return hits / n_trials


def summarize(name: str, interval: Interval, p_value: float | None = None) -> str:
    """One reportable line. Never a bare mean."""
    tail = "" if p_value is None else f"  p={p_value:.4g}"
    verdict = "significant" if interval.excludes_zero else "not significant"
    return f"{name:<28} {interval}  ({verdict}){tail}"
