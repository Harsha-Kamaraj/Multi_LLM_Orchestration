"""Turning a score into a probability, and proving it is one.

`P_pass` is not a ranking. The decision rule subtracts `lam * E_cost` from it,
so the two have to live on the same scale for λ to mean anything: at λ = 0.01
"one point of pass probability is worth 100 GPU-seconds" is a sentence about
*probabilities*, and it is false for a logistic score that happens to sort
correctly. An uncalibrated `P_pass` makes every λ on the frontier a different,
unknowable trade.

AUC cannot see this. A monotone transform of the scores leaves AUC exactly
where it was and can move ECE from 0.02 to 0.30, which is why the gate passing
says nothing about whether the policy's probabilities are usable.

## Isotonic, fitted on validation

Isotonic regression is the right shape here: it assumes only that a higher
score means a higher probability — which is the one thing a fitted classifier
does reliably — and otherwise takes its form from the data. Platt scaling would
impose a sigmoid on a distribution that has no reason to be one.

It is fitted on `val`, never on `train` (where the classifier's scores are
already optimistic, so the map learned there corrects the wrong distortion) and
structurally never on `test`, which `store.load_rollouts` cannot open.

## Why ECE is measured by cross-fitting

Fitting the calibrator on validation and then reporting its ECE on validation
measures how well isotonic fitted the noise. Isotonic is flexible enough to
drive that number near zero on any data at all, so "ECE < 0.05" earned that way
is a statement about the estimator, not about the probabilities.

So the honest number comes from cross-fitting: each validation row is scored by
a calibrator fitted on the *other* folds, and the shipped calibrator is then
fitted on all of validation. Both numbers are reported, because the gap between
them is itself informative — a large one means the calibrator is memorizing.

**The folds are grouped by task.** Three seeds of one task in three different
folds would put near-duplicates on both sides of the split, and the cross-fitted
number would inherit exactly the optimism it exists to remove. Same reason the
bootstrap clusters on task rather than on row.

## The artifact is knots, not a pickle

A fitted calibrator serializes as its breakpoints, and applying it is
`np.interp`. Nothing has to unpickle an estimator — or install scikit-learn —
to read what R3 shipped, and a reviewer can see the whole map as numbers. R4's
audit reads artifacts, and an artifact it cannot open without importing R3's
code is one it cannot independently check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import PolicyError

#: The definition of done in `docs/harsha.md`.
ECE_TARGET: float = 0.05

#: Bin count for ECE and the reliability table. Ten equal-width bins is the
#: convention the calibration literature reports against; it is fixed rather
#: than tuned because a bin count chosen after seeing the answer is a knob for
#: making ECE look good.
DEFAULT_BINS: int = 10


class CalibrationError(PolicyError):
    """A calibrator cannot be fitted or applied as asked."""


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityBin:
    """One row of a reliability diagram, kept as data rather than a plot.

    R4 owns figures. What R3 owes them is the table behind one, per arm, so the
    diagram can be drawn without re-deriving anything.
    """

    lower: float
    upper: float
    n: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        """Signed: positive means the model promised more than it delivered."""
        return self.mean_predicted - self.observed_rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "n": self.n,
            "mean_predicted": self.mean_predicted,
            "observed_rate": self.observed_rate,
            "gap": self.gap,
        }


def _checked(probabilities: Sequence[float] | np.ndarray,
             labels: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise CalibrationError(
            f"{p.size} probabilities against {y.size} labels; these must be "
            f"the same rows in the same order"
        )
    if p.size == 0:
        raise CalibrationError("no rows to calibrate")
    if not np.isfinite(p).all():
        raise CalibrationError(
            "probabilities contain NaN or infinity, which would silently "
            "become a bin of their own"
        )
    if (p < 0).any() or (p > 1).any():
        raise CalibrationError(
            f"probabilities must lie in [0, 1]; got [{p.min():.4f}, "
            f"{p.max():.4f}]. A raw decision-function score is not a "
            f"probability, and calibrating one as if it were hides that."
        )
    if not np.isin(y, (0.0, 1.0)).all():
        raise CalibrationError("labels must be 0 or 1")
    return p, y


def reliability_table(probabilities: Sequence[float] | np.ndarray,
                      labels: Sequence[int] | np.ndarray,
                      n_bins: int = DEFAULT_BINS) -> tuple[ReliabilityBin, ...]:
    """Predicted probability against observed frequency, in equal-width bins.

    Empty bins are dropped rather than reported as zero. A bin no row landed in
    carries no evidence, and plotting it at the origin draws a line through a
    point that was never measured.
    """
    p, y = _checked(probabilities, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `right=False` everywhere except the last bin, so p == 1.0 lands in the
    # top bin instead of a bin of its own.
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(ReliabilityBin(
            lower=float(edges[b]),
            upper=float(edges[b + 1]),
            n=count,
            mean_predicted=float(p[mask].mean()),
            observed_rate=float(y[mask].mean()),
        ))
    return tuple(bins)


def expected_calibration_error(probabilities: Sequence[float] | np.ndarray,
                               labels: Sequence[int] | np.ndarray,
                               n_bins: int = DEFAULT_BINS) -> float:
    """Mean absolute gap between promised and observed, weighted by bin size.

    Weighted by count, so a bin holding four rows cannot dominate one holding
    four hundred. This is the standard definition and it is deliberately not
    the maximum gap — R3 reports both, since ECE can look healthy while one
    sparse, confident bin is badly wrong.
    """
    table = reliability_table(probabilities, labels, n_bins)
    total = sum(b.n for b in table)
    return float(sum(b.n * abs(b.gap) for b in table) / total)


def maximum_calibration_error(probabilities: Sequence[float] | np.ndarray,
                              labels: Sequence[int] | np.ndarray,
                              n_bins: int = DEFAULT_BINS) -> float:
    """The worst bin. Reported beside ECE, never instead of it."""
    table = reliability_table(probabilities, labels, n_bins)
    return float(max(abs(b.gap) for b in table))


def brier_score(probabilities: Sequence[float] | np.ndarray,
                labels: Sequence[int] | np.ndarray) -> float:
    """Mean squared error of the probabilities.

    Kept because it is a *proper* scoring rule and ECE is not: ECE can be
    driven to zero by a model that ignores its input and predicts the base rate
    for every row. Brier falls only when the probabilities are both calibrated
    and informative, so the pair together catches what either alone misses.
    """
    p, y = _checked(probabilities, labels)
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# The fitted map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibrator:
    """A fitted isotonic map, stored as its breakpoints.

    Applying it is interpolation between knots, clamped at both ends. That is
    exactly what `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`
    does at predict time, and writing it out this way means the artifact can be
    read, diffed and applied by anything that has numpy.
    """

    knots_x: tuple[float, ...]
    knots_y: tuple[float, ...]
    fitted_on: tuple[str, ...]
    n_rows: int
    arm: str = ""

    def __post_init__(self) -> None:
        if len(self.knots_x) != len(self.knots_y):
            raise CalibrationError("knot arrays must agree in length")
        if not self.knots_x:
            raise CalibrationError("a calibrator needs at least one knot")
        if list(self.knots_x) != sorted(self.knots_x):
            raise CalibrationError("knots must be sorted by score")
        if list(self.knots_y) != sorted(self.knots_y):
            raise CalibrationError(
                "calibrated values must be non-decreasing; a map that inverts "
                "the ranking is not a calibration, it is a different model"
            )
        if "test" in self.fitted_on:
            raise CalibrationError(
                "a calibrator fitted on test is not a calibrator, it is a "
                "result. The test split is R4's and is opened once."
            )

    def __call__(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if p.size and ((p < 0).any() or (p > 1).any()):
            raise CalibrationError(
                "asked to calibrate values outside [0, 1]; the input to a "
                "calibrator is the classifier's probability, not its score"
            )
        out = np.interp(p, np.asarray(self.knots_x), np.asarray(self.knots_y))
        return np.clip(out, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "knots_x": list(self.knots_x),
            "knots_y": list(self.knots_y),
            "fitted_on": list(self.fitted_on),
            "n_rows": self.n_rows,
            "arm": self.arm,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Calibrator":
        return Calibrator(
            knots_x=tuple(float(v) for v in payload["knots_x"]),
            knots_y=tuple(float(v) for v in payload["knots_y"]),
            fitted_on=tuple(str(s) for s in payload["fitted_on"]),
            n_rows=int(payload["n_rows"]),
            arm=str(payload.get("arm", "")),
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                        encoding="utf-8", newline="")
        return path

    @staticmethod
    def load(path: Path | str) -> "Calibrator":
        return Calibrator.from_dict(json.loads(Path(path).read_text()))


def fit_calibrator(probabilities: Sequence[float] | np.ndarray,
                   labels: Sequence[int] | np.ndarray, *,
                   fitted_on: Sequence[str] = ("val",),
                   arm: str = "") -> Calibrator:
    """Fit isotonic regression, and refuse the cases where it means nothing."""
    p, y = _checked(probabilities, labels)
    if "test" in fitted_on:
        raise CalibrationError(
            "refusing to fit a calibrator on test. It is R4's split, opened "
            "once, after pre-registration."
        )
    if len(set(y.tolist())) < 2:
        raise CalibrationError(
            "the calibration split has one outcome class, so every score maps "
            "to the same probability and the map carries no information"
        )
    if np.ptp(p) == 0:
        raise CalibrationError(
            "every score is identical, so there is no ranking to calibrate. "
            "This is usually a degenerate model rather than a calibration "
            "problem."
        )

    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    return Calibrator(
        knots_x=tuple(float(v) for v in np.asarray(iso.X_thresholds_)),
        knots_y=tuple(float(v) for v in np.asarray(iso.y_thresholds_)),
        fitted_on=tuple(fitted_on),
        n_rows=int(p.size),
        arm=arm,
    )


def cross_fitted_probabilities(probabilities: Sequence[float] | np.ndarray,
                               labels: Sequence[int] | np.ndarray,
                               groups: Sequence[str], *,
                               n_folds: int = 5,
                               seed: int = 0) -> np.ndarray:
    """Calibrate every row with a map fitted on the other folds.

    `groups` is the task id. Folds are assigned per *task*, so no seed of a task
    is ever calibrated by a map that saw another seed of the same task.

    A fold whose training side is degenerate — one class, or no spread in the
    scores — falls back to the uncalibrated probabilities for that fold rather
    than raising. The alternative is that one unlucky fold costs the whole
    honest ECE number, and a fold-level fallback is visible in the result
    (it moves ECE toward the uncalibrated value) rather than silent.
    """
    p, y = _checked(probabilities, labels)
    groups = [str(g) for g in groups]
    if len(groups) != p.size:
        raise CalibrationError(
            f"{len(groups)} groups against {p.size} rows; the group is the "
            f"task and there is one per row"
        )

    unique = sorted(set(groups))
    if len(unique) < n_folds:
        raise CalibrationError(
            f"{len(unique)} tasks cannot be split into {n_folds} folds. The "
            f"fold is over tasks, not rows, because seeds of one task are not "
            f"independent observations."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    fold_of = {unique[int(t)]: i % n_folds for i, t in enumerate(order)}
    assignment = np.array([fold_of[g] for g in groups], dtype=int)

    out = np.array(p, dtype=float, copy=True)
    for fold in range(n_folds):
        held = assignment == fold
        rest = ~held
        if not held.any() or not rest.any():
            continue
        try:
            calibrator = fit_calibrator(p[rest], y[rest], fitted_on=("val",))
        except CalibrationError:
            continue  # documented above: leave this fold uncalibrated
        out[held] = calibrator(p[held])
    return out


# ---------------------------------------------------------------------------
# The reported result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationReport:
    """What calibration achieved, per arm, with the honest number named."""

    arm: str
    n_rows: int
    n_tasks: int
    ece_before: float
    ece_after: float
    ece_cross_fitted: float
    mce_cross_fitted: float
    brier_before: float
    brier_after: float
    base_rate: float
    bins: tuple[ReliabilityBin, ...]

    @property
    def meets_target(self) -> bool:
        """Judged on the cross-fitted number, which is the defensible one."""
        return self.ece_cross_fitted < ECE_TARGET

    @property
    def optimism(self) -> float:
        """How much fitting and scoring on the same rows flattered the result.

        Positive, and usually most of `ece_after`: isotonic drives the in-sample
        number to roughly zero whatever the data, so this is close to the whole
        of the honest number.
        """
        return self.ece_cross_fitted - self.ece_after

    def summary(self) -> str:
        verdict = "MEETS" if self.meets_target else "MISSES"
        return "\n".join([
            f"[{verdict}] arm {self.arm!r}: ECE {self.ece_cross_fitted:.4f} "
            f"(cross-fitted) vs target {ECE_TARGET:.2f}",
            f"    uncalibrated {self.ece_before:.4f} -> in-sample "
            f"{self.ece_after:.4f} -> cross-fitted "
            f"{self.ece_cross_fitted:.4f}",
            f"    worst bin {self.mce_cross_fitted:.4f}; Brier "
            f"{self.brier_before:.4f} -> {self.brier_after:.4f}",
            f"    {self.n_rows} rows over {self.n_tasks} tasks, base rate "
            f"{self.base_rate:.3f}",
        ])

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n_rows": self.n_rows,
            "n_tasks": self.n_tasks,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "ece_cross_fitted": self.ece_cross_fitted,
            "mce_cross_fitted": self.mce_cross_fitted,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "base_rate": self.base_rate,
            "meets_target": self.meets_target,
            "optimism": self.optimism,
            "ece_target": ECE_TARGET,
            "bins": [b.as_dict() for b in self.bins],
        }


def calibrate(probabilities: Sequence[float] | np.ndarray,
              labels: Sequence[int] | np.ndarray,
              groups: Sequence[str], *,
              arm: str = "",
              n_folds: int = 5,
              n_bins: int = DEFAULT_BINS,
              seed: int = 0) -> tuple[Calibrator, CalibrationReport]:
    """Fit the shipped calibrator and measure it honestly, in one call.

    The returned calibrator is fitted on every row given. The report's headline
    number is the cross-fitted one, so the artifact uses all the data and the
    claim about it does not.
    """
    p, y = _checked(probabilities, labels)
    calibrator = fit_calibrator(p, y, arm=arm)
    in_sample = calibrator(p)
    cross = cross_fitted_probabilities(p, y, groups, n_folds=n_folds, seed=seed)

    return calibrator, CalibrationReport(
        arm=arm,
        n_rows=int(p.size),
        n_tasks=len(set(str(g) for g in groups)),
        ece_before=expected_calibration_error(p, y, n_bins),
        ece_after=expected_calibration_error(in_sample, y, n_bins),
        ece_cross_fitted=expected_calibration_error(cross, y, n_bins),
        mce_cross_fitted=maximum_calibration_error(cross, y, n_bins),
        brier_before=brier_score(p, y),
        brier_after=brier_score(cross, y),
        base_rate=float(y.mean()),
        bins=reliability_table(cross, y, n_bins),
    )
