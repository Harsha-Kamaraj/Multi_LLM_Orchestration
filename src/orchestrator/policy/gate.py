"""The Phase 0 gate: `AUC_D0` and `AUC_D1`, each with an interval.

This is the thinnest path to the number the whole project is conditional on.
No calibration, no λ sweep, no serialized artifact — one classifier per
decision point, fitted on train, scored on validation, with a cluster bootstrap
around the result. Everything else in R3's plan is downstream of this answering
yes.

    | quantity | threshold | if it fails                                    |
    |----------|-----------|------------------------------------------------|
    | AUC_D1   | >= 0.75   | hard stop. Neither decision point has signal.  |
    | AUC_D0   | >= 0.65   | pre-generation routing is dead. Move to D1.    |

**Expect D0 to be weak.** 0.60–0.68 is the predicted band, and that is the
structure of the problem rather than a modelling failure: predicting whether a
model will solve a task, from the task alone, is genuinely hard. The asymmetry
against D1 is the most interesting number in the project, so a D0 model tuned
until it looks respectable would destroy the finding. Nothing here is tuned.

## Why the interval is not optional

Three seeds of one task are not three independent observations. A bootstrap
over rows treats them as such and returns an interval far too narrow — the
single easiest way to ship a confidently wrong result. So the resampling is
R4's `cluster_bootstrap`: tasks drawn with replacement, seeds drawn with
replacement *within* each chosen task.

R4's implementation is used rather than a second one written here. Two
implementations of the most delicate statistic in the project is how they
quietly disagree, and the number R3 reports at the gate should be computed by
the same estimator that R4 will verify it with.

`cluster_bootstrap` resamples a single `(n_tasks, n_seeds)` matrix, but AUC
needs scores *and* labels. So the matrix holds **row indices**, and the
statistic looks both up. The resampling logic stays R4's, unmodified.

## A verdict is not a boolean

The gate is stated as a threshold on a quantity, and a quantity estimated from
finite data has an interval around it. An estimate of 0.76 whose interval runs
[0.71, 0.81] has not cleared 0.75 in any sense worth acting on — it has failed
to answer the question. So `GateResult.verdict` is one of PASS, FAIL, or
INCONCLUSIVE, and the third is a real outcome that means *collect more data*,
not *round in your favour*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from eval.stats import Interval, cluster_bootstrap

from .errors import PolicyError
from .features import FeatureSet, feature_set, fit_standardizer
from .store import RolloutData

#: ROADMAP.md's Phase 0 gate. `AUC_D1` is the hard stop for the whole project.
THRESHOLDS: dict[str, float] = {"D0": 0.65, "D1": 0.75}

#: What the roadmap predicts, so a wildly different number reads as a bug
#: rather than as a finding. Reported alongside, never enforced.
EXPECTED_BANDS: dict[str, tuple[float, float]] = {
    "D0": (0.60, 0.68),
    "D1": (0.80, 0.90),
}


class GateError(PolicyError):
    """The gate cannot be measured as asked."""


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC, ties averaged.

    Ties are averaged so a constant score returns 0.5 rather than 1.0. A model
    that learned nothing must score as having learned nothing — an AUC of 1.0
    from a degenerate predictor is the most flattering possible way to be
    wrong.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class GateResult:
    """One decision point's answer, with everything needed to read it."""

    decision_point: str
    auc: Interval
    threshold: float
    expected_band: tuple[float, float]
    n_train_rows: int
    n_eval_rows: int
    n_eval_tasks: int
    n_seeds: int
    n_features: int
    base_rate: float
    arm: str
    run_id: str

    @property
    def verdict(self) -> str:
        """PASS, FAIL, or INCONCLUSIVE — the third is a real answer."""
        if self.auc.low >= self.threshold:
            return "PASS"
        if self.auc.high < self.threshold:
            return "FAIL"
        return "INCONCLUSIVE"

    @property
    def in_expected_band(self) -> bool:
        low, high = self.expected_band
        return low <= self.auc.point <= high

    def summary(self) -> str:
        low, high = self.expected_band
        lines = [
            f"[{self.verdict}] AUC_{self.decision_point} = {self.auc.point:.4f} "
            f"[{self.auc.low:.4f}, {self.auc.high:.4f}] "
            f"vs threshold {self.threshold:.2f}",
            f"    run {self.run_id}, arm {self.arm!r}, "
            f"{self.n_features} features",
            f"    fitted on {self.n_train_rows} train rows, scored on "
            f"{self.n_eval_rows} val rows "
            f"({self.n_eval_tasks} tasks x {self.n_seeds} seeds)",
            f"    base rate {self.base_rate:.3f}; roadmap expects "
            f"{low:.2f}-{high:.2f}"
            + ("" if self.in_expected_band else "  <- outside the expected band"),
        ]
        if self.verdict == "INCONCLUSIVE":
            lines.append(
                "    the interval straddles the threshold: this is not a pass, "
                "it is too little data to say"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_point": self.decision_point,
            "auc": self.auc.as_dict(),
            "threshold": self.threshold,
            "expected_band": list(self.expected_band),
            "verdict": self.verdict,
            "in_expected_band": self.in_expected_band,
            "n_train_rows": self.n_train_rows,
            "n_eval_rows": self.n_eval_rows,
            "n_eval_tasks": self.n_eval_tasks,
            "n_seeds": self.n_seeds,
            "n_features": self.n_features,
            "base_rate": self.base_rate,
            "arm": self.arm,
            "run_id": self.run_id,
        }


def cheap_arm(data: RolloutData) -> str:
    """Which arm the gate predicts the success of.

    Both gate numbers are about *"will the small model solve this"*, so the
    cheap arm has to be identified rather than assumed. Cost decides it when a
    costing is pinned, because "small" means "cheap" and nothing else; the name
    is only a fallback, since an arm names a role and the role naming is a
    convention the schema does not enforce.
    """
    arms = sorted(set(data.arms))
    if len(arms) < 2:
        raise GateError(
            f"run {data.run_id} has arms {arms}; the gate compares the cheap "
            f"arm against the expensive one and needs both"
        )

    if data.has_cost:
        totals: dict[str, list[float]] = {arm: [] for arm in arms}
        for row in data.rows:
            value = row.get("gpu_seconds")
            if value is not None:
                totals[str(row["arm"])].append(float(value))
        means = {arm: float(np.mean(v)) for arm, v in totals.items() if v}
        if len(means) == len(arms):
            return min(means, key=means.__getitem__)

    named = [arm for arm in arms if "small" in arm.lower()]
    if len(named) == 1:
        return named[0]

    raise GateError(
        f"cannot tell which of {arms} is the cheap arm. Pin a cost fingerprint "
        f"so the answer comes from measured cost, or name the arm explicitly — "
        f"guessing here would silently invert every gate number."
    )


def _index_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, int, int]:
    """Lay row indices out as `(n_tasks, n_seeds)` for the cluster bootstrap.

    Missing cells are NaN rather than repeated or dropped. A task swept with
    two seeds where others have three is a real thing; padding it by repeating
    a seed would weight that task's observed outcome twice, and dropping the
    task would bias toward whatever finished first.
    """
    by_task: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_task.setdefault(str(row["task_id"]), []).append(i)

    n_tasks = len(by_task)
    if n_tasks < 2:
        raise GateError(
            f"cluster bootstrap needs at least 2 tasks, got {n_tasks}. The "
            f"cluster is the task, so a single task has no variance to resample."
        )
    n_seeds = max(len(v) for v in by_task.values())

    matrix = np.full((n_tasks, n_seeds), np.nan, dtype=float)
    for r, (_, indices) in enumerate(sorted(by_task.items())):
        matrix[r, :len(indices)] = indices
    return matrix, n_tasks, n_seeds


def _fit_predict(train_rows, eval_rows, features: FeatureSet,
                 labels_of) -> tuple[np.ndarray, int]:
    """Fit logistic regression on train, score validation. Nothing is tuned.

    Logistic regression, deliberately. "We tried the simple thing first and
    here is the number" is a stronger result than a complicated one that cannot
    be attributed, and the gate is a question about the *data*, not about how
    hard R3 can push an estimator.
    """
    from sklearn.linear_model import LogisticRegression

    train_matrix = features.build(train_rows)
    eval_matrix = features.build(eval_rows)

    # Fitted on train alone — including the mean and standard deviation, which
    # is the leak that looks identical to the correct version in a notebook.
    scaler = fit_standardizer(train_matrix, train_rows, on=("train",))
    X_train = scaler.transform(train_matrix).X
    X_eval = scaler.transform(eval_matrix).X

    y_train = np.array([labels_of(row) for row in train_rows], dtype=int)
    if len(set(y_train.tolist())) < 2:
        raise GateError(
            "the training split has only one outcome class, so nothing "
            "separates. Either every attempt succeeded or every one failed."
        )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1], X_train.shape[1]


def measure_gate(data: RolloutData, decision_point: str, *,
                 arm: str | None = None,
                 features: FeatureSet | None = None,
                 n_resamples: int = 2000,
                 level: float = 0.95,
                 seed: int = 0) -> GateResult:
    """Measure one decision point's AUC, with a clustered interval.

    Fitted on `train`, scored on `val`. The test split is not reachable from
    here — `load_rollouts` cannot load it — so there is nothing to opt out of.
    """
    target = arm or cheap_arm(data)
    features = features or feature_set(decision_point)

    rows = [row for row in data.rows if str(row["arm"]) == target]
    if not rows:
        raise GateError(
            f"no rows for arm {target!r}; run {data.run_id} has {list(data.arms)}"
        )

    train_rows = [r for r in rows if str(r.get("split")) == "train"]
    eval_rows = [r for r in rows if str(r.get("split")) == "val"]
    for name, subset in (("train", train_rows), ("val", eval_rows)):
        if not subset:
            raise GateError(
                f"no {name} rows for arm {target!r}. The gate fits on train and "
                f"scores on val; a run with only one of them cannot produce an "
                f"honest number."
            )

    def solved(row: Mapping[str, Any]) -> int:
        return int(data.label_for(str(row["rollout_id"])).solved)

    scores, n_features = _fit_predict(train_rows, eval_rows, features, solved)
    labels = np.array([solved(row) for row in eval_rows], dtype=int)

    matrix, n_tasks, n_seeds = _index_matrix(eval_rows)

    def auc_of(sample: np.ndarray) -> float:
        # `sample` holds row indices. NaN marks a task-seed cell that does not
        # exist, and is dropped rather than imputed.
        flat = sample.ravel()
        idx = flat[~np.isnan(flat)].astype(int)
        return auc(scores[idx], labels[idx])

    interval = cluster_bootstrap(
        matrix, auc_of, n_resamples=n_resamples, level=level, seed=seed,
    )

    return GateResult(
        decision_point=decision_point,
        auc=interval,
        threshold=THRESHOLDS[decision_point],
        expected_band=EXPECTED_BANDS[decision_point],
        n_train_rows=len(train_rows),
        n_eval_rows=len(eval_rows),
        n_eval_tasks=n_tasks,
        n_seeds=n_seeds,
        n_features=n_features,
        base_rate=float(labels.mean()),
        arm=target,
        run_id=data.run_id,
    )


def measure_gates(data: RolloutData, **kwargs: Any) -> dict[str, GateResult]:
    """Both decision points. The comparison between them is the finding."""
    return {point: measure_gate(data, point, **kwargs)
            for point in ("D0", "D1")}


def gate_report(results: Mapping[str, GateResult]) -> str:
    """Human-readable verdict, including the asymmetry."""
    lines = [result.summary() for result in results.values()]
    if "D0" in results and "D1" in results:
        d0, d1 = results["D0"].auc.point, results["D1"].auc.point
        lines.append(
            f"\nD1 - D0 = {d1 - d0:+.4f}. Observing failure beats predicting "
            f"it, and the size of that gap is the most interesting number in "
            f"the project."
        )
        if results["D1"].verdict == "FAIL":
            lines.append(
                "\nAUC_D1 is below 0.75. That is ROADMAP.md's hard stop: "
                "neither decision point carries signal and the premise is "
                "false. This is a finding, and it is reported rather than tuned."
            )
    return "\n".join(lines)


def write_gate_report(results: Mapping[str, GateResult], path: Path | str,
                      **extra: Any) -> Path:
    """Persist the gate numbers next to the run they were measured from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        point: result.as_dict() for point, result in results.items()
    }
    payload.update(extra)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="",
    )
    return path
