"""Features that declare their decision point, and cannot exceed it.

Every feature must be computable at the decision point it claims to serve. The
usual way to enforce that is a review comment and a hope. This module makes it
mechanical, in two layers that catch different mistakes:

**Declaration is checked against the contract.** A feature names the columns it
reads. At construction those are intersected with `contract.observable_at`, and
a D0 feature naming a D1 column fails before any data is loaded. This catches
the honest mistake — someone genuinely believing `visible_passed` is available
before generation.

**Access is restricted to the declaration.** A feature is not handed the row.
It is handed a `RowView` that raises on any key it did not declare, so a
feature whose declaration says one thing and whose body does another fails on
the first row rather than producing a number. This catches the mistake that
actually happens — a feature edited later to read one more column, with the
declaration left alone.

The second layer is why the declaration is trustworthy. Without it,
`source_columns` is documentation, and documentation drifts.

## What a feature may not read, ever

`contract.NEVER_A_FEATURE` is rejected at construction regardless of decision
point: the hidden-test columns, and `wall_ms`, which under `mode == "sweep"` is
a queue-depth measurement rather than anything about the model.

The loader has already removed the hidden columns from the rows by this point.
The check here is the second of the two independent guards the leakage
docstring in `contract.py` describes — a guard that runs in only one place
stops being a guard the moment someone adds a second path.

## Siblings, and why they are not free

A few features read the *other* draws of the same task and arm — how often the
visible tests agreed across seeds is a real difficulty signal. Two rules apply
and both are enforced here rather than trusted:

* **A row is never its own sibling.** Including it leaks the outcome being
  predicted into the prediction, and the resulting AUC looks excellent.
* **Sibling features are paid for.** A policy using k draws must be charged for
  k draws, or the matched-cost comparison it feeds is fiction. Such features
  declare `paid_arms`, and nothing in the default D1 set uses them — the probe
  is a Phase 2 deliverable and turning it on is a deliberate act.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .. import contract
from ..errors import LeakageError, PolicyError

#: The two decision points. There is no third, and `contract.observable_at`
#: refuses anything else.
DECISION_POINTS: tuple[str, ...] = ("D0", "D1")


class FeatureError(PolicyError):
    """A feature is declared or used in a way the contract does not allow."""


class RowView(Mapping[str, Any]):
    """A row restricted to one feature's declared columns.

    Raises rather than returning `None` for an undeclared column. Returning
    `None` would let a feature silently read nothing and emit a plausible
    constant, which is the failure this class exists to make impossible.
    """

    __slots__ = ("_row", "_allowed", "_feature")

    def __init__(self, row: Mapping[str, Any], allowed: Iterable[str],
                 feature: str) -> None:
        self._row = row
        self._allowed = frozenset(allowed)
        self._feature = feature

    def __getitem__(self, key: str) -> Any:
        if key not in self._allowed:
            raise LeakageError(
                f"feature {self._feature!r} read column {key!r}, which it did "
                f"not declare. Declared: {sorted(self._allowed)}. Add it to "
                f"`source_columns` if it is legitimately readable at this "
                f"decision point — the declaration is what R4's leakage audit "
                f"reads, so a feature that quietly reads more than it declares "
                f"makes that audit wrong rather than merely incomplete."
            )
        return self._row.get(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._allowed)

    def __len__(self) -> int:
        return len(self._allowed)

    def __repr__(self) -> str:
        return f"RowView({self._feature!r}, {sorted(self._allowed)})"


@dataclass(frozen=True)
class Feature:
    """One scalar, and the exact evidence it is allowed to be computed from."""

    name: str
    decision_point: str
    source_columns: tuple[str, ...]
    fn: Callable[..., float]
    description: str = ""
    #: True when `fn` takes `(view, siblings)` rather than `(view,)`.
    needs_siblings: bool = False
    #: Arms whose generation this feature obliges the policy to pay for. A
    #: non-empty value here is a claim about cost accounting, not a label.
    paid_arms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.decision_point not in DECISION_POINTS:
            raise FeatureError(
                f"feature {self.name!r} claims decision point "
                f"{self.decision_point!r}; this project has exactly "
                f"{list(DECISION_POINTS)}"
            )
        if not self.source_columns:
            raise FeatureError(
                f"feature {self.name!r} declares no source columns. A feature "
                f"that reads nothing is a constant, and a constant column "
                f"trains nothing while looking like evidence."
            )

        forbidden = sorted(set(self.source_columns) & contract.NEVER_A_FEATURE)
        if forbidden:
            raise LeakageError(
                f"feature {self.name!r} declares {forbidden}, which are never "
                f"features at any decision point. Hidden-test columns are "
                f"labels; `wall_ms` under mode=sweep is a queue-depth "
                f"measurement rather than a property of the model."
            )

        observable = contract.observable_at(self.decision_point)
        premature = sorted(set(self.source_columns) - observable)
        if premature:
            later = sorted(set(premature) & contract.D1_OBSERVABLE)
            hint = (
                f" {later} exist only after generating, so a feature reading "
                f"them is a D1 feature."
                if later and self.decision_point == "D0" else ""
            )
            raise LeakageError(
                f"feature {self.name!r} claims {self.decision_point} but "
                f"declares {premature}, which is not observable there.{hint}"
            )

        if self.paid_arms and not self.needs_siblings:
            raise FeatureError(
                f"feature {self.name!r} declares paid_arms {list(self.paid_arms)} "
                f"but reads no siblings; a feature computed from the row it "
                f"describes costs nothing extra to obtain"
            )

    def compute(self, row: Mapping[str, Any],
                siblings: Sequence[Mapping[str, Any]] = ()) -> float:
        view = RowView(row, self.source_columns, self.name)
        if not self.needs_siblings:
            return float(self.fn(view))
        views = tuple(
            RowView(sibling, self.source_columns, self.name)
            for sibling in siblings
        )
        return float(self.fn(view, views))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decision_point": self.decision_point,
            "source_columns": list(self.source_columns),
            "description": self.description,
            "needs_siblings": self.needs_siblings,
            "paid_arms": list(self.paid_arms),
        }


@dataclass(frozen=True)
class FeatureMatrix:
    """A built design matrix, still keyed to the rows it came from."""

    X: np.ndarray
    names: tuple[str, ...]
    rollout_ids: tuple[str, ...]
    decision_point: str

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.X.shape[0]), int(self.X.shape[1]))

    def column(self, name: str) -> np.ndarray:
        try:
            return self.X[:, self.names.index(name)]
        except ValueError:
            raise FeatureError(
                f"no feature named {name!r}; built features are "
                f"{list(self.names)}"
            ) from None

    def constant_columns(self) -> tuple[str, ...]:
        """Features with no variance, which train nothing.

        Not an error — a keyword flag can legitimately be absent from a small
        fixture. Reported so it is a known fact rather than a silent one.
        """
        return tuple(
            name for i, name in enumerate(self.names)
            if float(np.ptp(self.X[:, i])) == 0.0
        )


class FeatureSet:
    """An ordered, validated collection of features for one decision point."""

    def __init__(self, features: Sequence[Feature], *, name: str = "") -> None:
        seen: set[str] = set()
        for feature in features:
            if feature.name in seen:
                raise FeatureError(
                    f"duplicate feature name {feature.name!r}; names index "
                    f"columns of the design matrix and must be unique"
                )
            seen.add(feature.name)
        points = {f.decision_point for f in features}
        if len(points) > 1:
            raise FeatureError(
                f"a feature set spans decision points {sorted(points)}. Build "
                f"one set per decision point — mixing them is how a D1 column "
                f"reaches a D0 model."
            )
        self.features = tuple(features)
        self.name = name
        self.decision_point = points.pop() if points else "D0"

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.features)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Every column this set reads. What R4's leakage audit checks."""
        columns: set[str] = set()
        for feature in self.features:
            columns.update(feature.source_columns)
        return tuple(sorted(columns))

    @property
    def paid_arms(self) -> tuple[str, ...]:
        arms: set[str] = set()
        for feature in self.features:
            arms.update(feature.paid_arms)
        return tuple(sorted(arms))

    def __add__(self, other: "FeatureSet") -> "FeatureSet":
        return FeatureSet(self.features + other.features,
                          name=f"{self.name}+{other.name}".strip("+"))

    def build(self, rows: Sequence[Mapping[str, Any]]) -> FeatureMatrix:
        """Compute the design matrix for `rows`.

        Sibling groups are keyed on `(task_id, arm)` and exclude the row
        itself, so a sibling feature can never read the outcome it is being
        used to predict.
        """
        if not rows:
            raise FeatureError("no rows to build features from")

        missing = sorted(set(self.source_columns) - set(rows[0]))
        if missing:
            raise FeatureError(
                f"feature set {self.name or self.decision_point!r} needs "
                f"columns {missing}, which are absent from the loaded rows. "
                f"Prompt-side columns come from R2's task manifest — pass "
                f"`tasks_path` to `load_rollouts`, since the rollout row "
                f"carries no prompt."
            )

        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        if any(f.needs_siblings for f in self.features):
            for row in rows:
                key = (str(row.get("task_id")), str(row.get("arm")))
                groups.setdefault(key, []).append(row)

        X = np.empty((len(rows), len(self.features)), dtype=float)
        for r, row in enumerate(rows):
            siblings: tuple[Mapping[str, Any], ...] = ()
            if groups:
                key = (str(row.get("task_id")), str(row.get("arm")))
                # Identity comparison, not equality: two draws of one cell can
                # be equal dicts, and dropping both would silently shrink the
                # sibling set exactly where seeds agreed.
                siblings = tuple(s for s in groups[key] if s is not row)
            for c, feature in enumerate(self.features):
                X[r, c] = feature.compute(row, siblings)

        if not np.isfinite(X).all():
            bad = sorted({
                self.features[c].name
                for c in np.unique(np.argwhere(~np.isfinite(X))[:, 1])
            })
            raise FeatureError(
                f"features {bad} produced NaN or infinity. A non-finite "
                f"feature silently drops rows in most estimators, so the model "
                f"trains on a subset nobody chose."
            )

        return FeatureMatrix(
            X=X,
            names=self.names,
            rollout_ids=tuple(str(row["rollout_id"]) for row in rows),
            decision_point=self.decision_point,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decision_point": self.decision_point,
            "n_features": len(self.features),
            "source_columns": list(self.source_columns),
            "paid_arms": list(self.paid_arms),
            "features": [f.to_dict() for f in self.features],
        }

    def write_spec(self, path: Path | str, **extra: Any) -> Path:
        """Write `feature_spec.json` — the file R4's leakage audit reads.

        Written separately from the model artifact on purpose: the audit
        should be runnable against the declaration without unpickling anything
        R3 produced.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload.update(extra)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="",
        )
        return path


def feature(name: str, decision_point: str, *source_columns: str,
            description: str = "", needs_siblings: bool = False,
            paid_arms: tuple[str, ...] = ()) -> Callable[[Callable], Feature]:
    """Decorator form, so a feature's columns sit next to its body.

    Keeping the declaration adjacent to the code is the whole reason the two
    stay in step; a registry in another file drifts from what the functions
    actually read.
    """

    def wrap(fn: Callable[..., float]) -> Feature:
        return Feature(
            name=name,
            decision_point=decision_point,
            source_columns=tuple(source_columns),
            fn=fn,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            needs_siblings=needs_siblings,
            paid_arms=paid_arms,
        )

    return wrap
