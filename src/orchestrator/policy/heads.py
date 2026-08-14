"""The three value heads, fitted per arm.

    P_pass[a](x)     calibrated probability arm a solves the task
    E_cost[a](x)     expected GPU-seconds, via predicted tokens
    E_latency[a](x)  expected wall-clock seconds, via the same tokens

Three heads rather than one model of `accuracy - lam*cost - lam2*latency`. The
scalarized version bakes λ into the weights, so every new trade-off is a
retrain and the output is a point rather than a frontier. Here λ multiplies two
predictions that were fitted without ever having met it, one training run
serves the whole curve, and λ becomes a dial someone can turn in production
without R3 present.

## Arm identity indexes the head; it is not a feature

`P_pass` is fitted separately per arm. Pooling the arms and adding a one-hot
column would force one set of feature weights to describe both models, so a
prompt feature that predicts success for the 1.5B and predicts nothing for the
7B has to compromise into a single coefficient. Separate fits let the arms
disagree about what makes a task hard, which is most of what the policy is for.

## E_cost predicts tokens, never dollars

The head's target is `prefill_tokens` and `decode_tokens`. Conversion to
GPU-seconds, latency and USD happens at *scoring* time, through R1's pinned
`CostCoefficients`.

Training against dollars directly would bake one hardware rate and one instance
price into the weights, and both change without the model being wrong — a spot
price moves and every prediction is silently stale, with no way to tell that
from a genuine drift in behaviour. Tokens are a property of the model and the
prompt. Prices are a property of a contract with a cloud vendor.

The conversion is R1's function rather than a second implementation of the same
arithmetic here, for the same reason the gate calls R4's bootstrap: the number
R3 reports and the number R1 measured should come from one place. R1 already
draws the distinction that matters — `gpu_seconds` is work-proportional
occupancy with the fixed per-request cost removed, while `imputed_latency_s`
includes it, because a user waits through the intercept and a GPU is not
occupied by it.

## What is fitted where

Everything is fitted on `train`: feature standardization, the classifier, and
both token regressions. Only the calibrator is fitted on `val`, which is what
`val` is for — a classifier's training-set scores are already optimistic, so a
map learned there would correct a distortion that does not exist at inference.

`test` is not reachable. `store.load_rollouts` will not open it.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import (
    Calibrator, CalibrationReport, calibrate,
)
from .errors import PolicyError, SplitError
from .features import FeatureSet, Standardizer, feature_set, fit_standardizer
from .store import RolloutData

#: Targets of the two cost heads. D1-observable columns, read here as labels
#: rather than as inputs — at D0 they are exactly what is being predicted.
TOKEN_TARGETS: tuple[str, ...] = ("prefill_tokens", "decode_tokens")

#: Filenames of the artifact set. `policy.pkl` holds the estimators; everything
#: needed to audit the policy is in the JSON beside it, so a leakage review
#: never has to unpickle anything R3 produced.
POLICY_PICKLE = "policy.pkl"
HEADS_MANIFEST = "heads.json"
CALIBRATION_FILE = "calibration.json"
FEATURE_SPEC_FILE = "feature_spec.json"


class HeadError(PolicyError):
    """The heads cannot be fitted or applied as asked."""


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------


@dataclass
class ArmHeads:
    """All three heads for a single arm, plus the scaler they were fitted with.

    The scaler travels with the estimators deliberately. A head reloaded
    without the constants it was fitted under would be handed raw features and
    would produce confident nonsense, and nothing about the numbers would look
    wrong.
    """

    arm: str
    model_id: str
    decision_point: str
    feature_names: tuple[str, ...]
    scaler: Standardizer
    pass_model: Any
    prefill_model: Any | None
    decode_model: Any | None
    calibrator: Calibrator | None = None
    n_train_rows: int = 0
    n_token_rows: int = 0

    # -- P_pass --------------------------------------------------------------

    def p_pass(self, rows: Sequence[Mapping[str, Any]], features: FeatureSet,
               *, calibrated: bool = True) -> np.ndarray:
        """Probability this arm solves each row's task.

        Calibrated by default. The uncalibrated path exists for measuring the
        difference, not for scoring — an uncalibrated `P_pass` is a ranking, and
        subtracting `lam * E_cost` from a ranking is meaningless.
        """
        raw = self.pass_model.predict_proba(self._design(rows, features))[:, 1]
        if not calibrated:
            return raw
        if self.calibrator is None:
            raise HeadError(
                f"arm {self.arm!r} has no calibrator, so there is no calibrated "
                f"probability to return. Fit one on val, or ask for "
                f"calibrated=False and accept that the number is a score."
            )
        return self.calibrator(raw)

    # -- tokens, and the three things they convert into ----------------------

    def tokens(self, rows: Sequence[Mapping[str, Any]],
               features: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
        """Predicted `(prefill_tokens, decode_tokens)`, clipped at zero.

        Negative token counts are the obvious failure of an unconstrained
        linear fit near the low end, and they would turn into negative costs
        that the decision rule reads as a *reward* for choosing an arm.
        """
        if self.prefill_model is None or self.decode_model is None:
            raise HeadError(
                f"arm {self.arm!r} has no token heads: the training rows "
                f"carried no {TOKEN_TARGETS} to fit them against. Cost cannot "
                f"be predicted for this arm."
            )
        X = self._design(rows, features)
        prefill = np.clip(self.prefill_model.predict(X), 0.0, None)
        decode = np.clip(self.decode_model.predict(X), 0.0, None)
        return prefill, decode

    def e_cost(self, rows, features: FeatureSet, coefficients) -> np.ndarray:
        """Expected GPU-seconds. Work-proportional occupancy, R1's definition."""
        prefill, decode = self.tokens(rows, features)
        return self._convert(coefficients.gpu_seconds, prefill, decode)

    def e_latency(self, rows, features: FeatureSet, coefficients) -> np.ndarray:
        """Expected wall-clock seconds, including the fixed per-request cost."""
        prefill, decode = self.tokens(rows, features)
        return self._convert(coefficients.imputed_latency_s, prefill, decode)

    def e_usd(self, rows, features: FeatureSet, coefficients) -> np.ndarray:
        """Dollars under the rate recorded in the coefficients, not a new one."""
        prefill, decode = self.tokens(rows, features)
        return self._convert(coefficients.usd, prefill, decode)

    def _convert(self, fn, prefill: np.ndarray, decode: np.ndarray) -> np.ndarray:
        try:
            return np.array([
                float(fn(self.model_id, float(p), float(d)))
                for p, d in zip(prefill, decode)
            ])
        except KeyError as exc:
            # R1 raises rather than falling back to another model's
            # coefficients, which is right — the fallback would produce cost
            # numbers that look fine and describe different weights. Restated
            # as a policy error so the CLI prints it instead of a traceback.
            raise HeadError(
                f"arm {self.arm!r} serves {self.model_id!r}, which the pinned "
                f"costing does not cover, so its cost cannot be converted. "
                f"Point --coefficients at a characterization that includes it. "
                f"Underlying: {exc}"
            ) from exc

    def _design(self, rows: Sequence[Mapping[str, Any]],
                features: FeatureSet) -> np.ndarray:
        matrix = features.build(rows)
        if matrix.names != self.feature_names:
            raise HeadError(
                f"arm {self.arm!r} was fitted on a different feature set. "
                f"Scoring with mismatched columns applies the wrong "
                f"coefficient to every one of them and raises nothing."
            )
        return self.scaler.transform(matrix).X

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "model_id": self.model_id,
            "decision_point": self.decision_point,
            "n_features": len(self.feature_names),
            "n_train_rows": self.n_train_rows,
            "n_token_rows": self.n_token_rows,
            "has_calibrator": self.calibrator is not None,
            "has_token_heads": self.prefill_model is not None,
            "scaler_fitted_on": list(self.scaler.fitted_on),
        }


# ---------------------------------------------------------------------------
# Every arm
# ---------------------------------------------------------------------------


@dataclass
class PolicyHeads:
    """One fitted policy: every arm's three heads, and where they came from."""

    run_id: str
    decision_point: str
    features: FeatureSet
    arms: dict[str, ArmHeads] = field(default_factory=dict)
    calibration: dict[str, CalibrationReport] = field(default_factory=dict)
    publishable: bool = False

    def __post_init__(self) -> None:
        if not self.arms:
            raise HeadError("a policy with no arms cannot choose anything")

    @property
    def arm_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.arms))

    @property
    def meets_calibration_target(self) -> bool:
        """Every arm, not the average. A policy is only as usable as its worst
        head — the decision rule compares arms against each other, so one
        badly calibrated arm corrupts every comparison it takes part in."""
        return bool(self.calibration) and all(
            report.meets_target for report in self.calibration.values()
        )

    def p_pass(self, arm: str, rows, **kwargs) -> np.ndarray:
        return self._arm(arm).p_pass(rows, self.features, **kwargs)

    def e_cost(self, arm: str, rows, coefficients) -> np.ndarray:
        return self._arm(arm).e_cost(rows, self.features, coefficients)

    def e_latency(self, arm: str, rows, coefficients) -> np.ndarray:
        return self._arm(arm).e_latency(rows, self.features, coefficients)

    def _arm(self, arm: str) -> ArmHeads:
        if arm not in self.arms:
            raise HeadError(
                f"no head for arm {arm!r}; this policy was fitted for "
                f"{list(self.arm_names)}"
            )
        return self.arms[arm]

    # -- artifacts -----------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decision_point": self.decision_point,
            "publishable": self.publishable,
            "feature_set": self.features.name,
            "n_features": len(self.features),
            "arms": {name: head.as_dict() for name, head in self.arms.items()},
            "calibration": {
                name: report.as_dict() for name, report in self.calibration.items()
            },
            "meets_calibration_target": self.meets_calibration_target,
        }

    def save(self, directory: Path | str) -> Path:
        """Write the artifact set: estimators pickled, everything else JSON.

        `decisions.parquet` is Phase 6 and is not written here — a policy and
        the decisions it produced at a given λ are separate artifacts, because
        the λ sweep re-reads one policy many times.

        ## The feature set is not pickled

        Only fitted *state* goes into the pickle — estimators, scaling
        constants, calibration knots. The `FeatureSet` is code, and pickling
        code freezes a copy of R3's feature builders inside an artifact where
        nobody will think to look at them.

        Instead the decision point is recorded and the feature set is rebuilt
        from it on load, then checked against the column names the heads were
        actually fitted with. So editing a feature builder and reloading an old
        policy *fails loudly* rather than scoring with new code under old
        coefficients — which is the failure this would otherwise cause, and it
        would look like a modelling result.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with (directory / POLICY_PICKLE).open("wb") as handle:
            pickle.dump({
                "run_id": self.run_id,
                "decision_point": self.decision_point,
                "publishable": self.publishable,
                "feature_names": self.features.names,
                "arms": self.arms,
                "calibration": self.calibration,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)

        (directory / HEADS_MANIFEST).write_text(
            json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="",
        )
        self.features.write_spec(
            directory / FEATURE_SPEC_FILE, run_id=self.run_id,
        )
        (directory / CALIBRATION_FILE).write_text(
            json.dumps(
                {name: head.calibrator.to_dict()
                 for name, head in self.arms.items()
                 if head.calibrator is not None},
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8", newline="",
        )
        return directory

    @staticmethod
    def load(directory: Path | str) -> "PolicyHeads":
        """Rebuild the policy, refusing one whose features have moved under it."""
        with (Path(directory) / POLICY_PICKLE).open("rb") as handle:
            payload = pickle.load(handle)

        features = feature_set(payload["decision_point"])
        recorded = tuple(payload["feature_names"])
        if features.names != recorded:
            extra = sorted(set(features.names) - set(recorded))
            missing = sorted(set(recorded) - set(features.names))
            raise HeadError(
                f"this policy was fitted against a different version of the "
                f"{payload['decision_point']} feature set. Added since: "
                f"{extra}. Gone: {missing}. Reordering counts too. Refit it — "
                f"scoring new features with old coefficients produces numbers "
                f"that look entirely reasonable."
            )
        return PolicyHeads(
            run_id=payload["run_id"],
            decision_point=payload["decision_point"],
            features=features,
            arms=payload["arms"],
            calibration=payload["calibration"],
            publishable=payload["publishable"],
        )

    def summary(self) -> str:
        lines = [
            f"policy for {self.decision_point} from run {self.run_id}"
            + ("" if self.publishable else "  [not publishable]"),
            f"  arms: {', '.join(self.arm_names)}",
        ]
        lines.extend("  " + line
                     for name in self.arm_names if name in self.calibration
                     for line in self.calibration[name].summary().splitlines())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _model_id_for(rows: Sequence[Mapping[str, Any]], arm: str) -> str:
    """The served model behind an arm, which is what a costing is keyed by.

    An arm whose rows disagree about `model_id` is refused rather than resolved
    by majority: it means two different models were swept under one arm name,
    and every cost number for that arm would be an average of two hardware
    profiles.
    """
    seen = {str(row.get("model_id")) for row in rows if row.get("model_id")}
    if not seen:
        raise HeadError(
            f"arm {arm!r} carries no model_id, so its cost cannot be converted "
            f"— a costing is keyed by model, not by arm name"
        )
    if len(seen) > 1:
        raise HeadError(
            f"arm {arm!r} maps to more than one model: {sorted(seen)}. One arm "
            f"is one served model; two would make every cost for this arm an "
            f"average over hardware profiles."
        )
    return seen.pop()


def _fit_token_head(X: np.ndarray, rows: Sequence[Mapping[str, Any]],
                    column: str):
    """Ridge on the rows that actually carry the target.

    Ridge rather than plain least squares because the feature set is wide,
    standardized, and contains near-duplicates by construction (prompt length
    in characters and in lines are close to collinear). A mild penalty keeps
    the coefficients readable; it is not tuned, and no amount of tuning it
    would change what the head is for.

    Returns `(model, n_rows)`, or `(None, 0)` when nothing carries the target —
    an ungraded or uncosted run should lose its cost heads, not its whole
    policy.
    """
    from sklearn.linear_model import Ridge

    mask = np.array([row.get(column) is not None for row in rows])
    if mask.sum() < 2:
        return None, 0
    y = np.array([float(row[column]) for row in rows if row.get(column) is not None])
    model = Ridge(alpha=1.0)
    model.fit(X[mask], y)
    return model, int(mask.sum())


def fit_arm(data: RolloutData, arm: str, decision_point: str, *,
            features: FeatureSet | None = None,
            n_folds: int = 5,
            seed: int = 0) -> tuple[ArmHeads, CalibrationReport | None]:
    """Fit one arm's three heads on train, and calibrate on val."""
    from sklearn.linear_model import LogisticRegression

    features = features or feature_set(decision_point)
    rows = [row for row in data.rows if str(row["arm"]) == arm]
    if not rows:
        raise HeadError(f"run {data.run_id} has no rows for arm {arm!r}")

    train_rows = [r for r in rows if str(r.get("split")) == "train"]
    val_rows = [r for r in rows if str(r.get("split")) == "val"]
    if not train_rows:
        raise SplitError(
            f"arm {arm!r} has no train rows. Everything except the calibrator "
            f"is fitted on train; there is nothing to fit."
        )

    train_matrix = features.build(train_rows)
    scaler = fit_standardizer(train_matrix, train_rows, on=("train",))
    X_train = scaler.transform(train_matrix).X

    y_train = np.array(
        [int(data.label_for(str(r["rollout_id"])).solved) for r in train_rows],
        dtype=int,
    )
    if len(set(y_train.tolist())) < 2:
        raise HeadError(
            f"arm {arm!r} has one outcome class on train, so `P_pass` would be "
            f"a constant. Either every attempt succeeded or every one failed."
        )

    pass_model = LogisticRegression(max_iter=2000)
    pass_model.fit(X_train, y_train)

    prefill_model, n_prefill = _fit_token_head(X_train, train_rows,
                                               "prefill_tokens")
    decode_model, n_decode = _fit_token_head(X_train, train_rows,
                                             "decode_tokens")
    if prefill_model is None or decode_model is None:
        prefill_model = decode_model = None

    head = ArmHeads(
        arm=arm,
        model_id=_model_id_for(rows, arm),
        decision_point=decision_point,
        feature_names=train_matrix.names,
        scaler=scaler,
        pass_model=pass_model,
        prefill_model=prefill_model,
        decode_model=decode_model,
        n_train_rows=len(train_rows),
        n_token_rows=min(n_prefill, n_decode),
    )

    if not val_rows:
        return head, None

    raw = head.p_pass(val_rows, features, calibrated=False)
    y_val = np.array(
        [int(data.label_for(str(r["rollout_id"])).solved) for r in val_rows],
        dtype=int,
    )
    groups = [str(row["task_id"]) for row in val_rows]
    calibrator, report = calibrate(raw, y_val, groups, arm=arm,
                                   n_folds=n_folds, seed=seed)
    head.calibrator = calibrator
    return head, report


def fit_heads(data: RolloutData, decision_point: str, *,
              features: FeatureSet | None = None,
              arms: Sequence[str] | None = None,
              n_folds: int = 5,
              seed: int = 0) -> PolicyHeads:
    """Fit every arm's heads. This is the Phase 5 deliverable.

    An arm that cannot be fitted is not silently dropped — the decision rule
    compares arms, so a policy missing one is a different policy, and one that
    quietly became a single-arm policy would still produce decisions.
    """
    features = features or feature_set(decision_point)
    targets = list(arms) if arms is not None else sorted(set(data.arms))
    if not targets:
        raise HeadError(f"run {data.run_id} has no arms")

    fitted: dict[str, ArmHeads] = {}
    reports: dict[str, CalibrationReport] = {}
    for arm in targets:
        head, report = fit_arm(data, arm, decision_point, features=features,
                               n_folds=n_folds, seed=seed)
        fitted[arm] = head
        if report is not None:
            reports[arm] = report

    return PolicyHeads(
        run_id=data.run_id,
        decision_point=decision_point,
        features=features,
        arms=fitted,
        calibration=reports,
        publishable=data.publishable,
    )
