"""Normalization constants, fitted once on train and carried everywhere after.

This is one of the leaks that actually happens, and it happens because the
convenient thing and the correct thing look identical in a notebook. Calling
`fit_transform` on the whole matrix standardizes every column using a mean and
a standard deviation computed partly from the rows being evaluated. Nothing
errors. The model is slightly better than it should be, by an amount nobody can
estimate afterwards.

So the constants are an artifact here, not a step. They are fitted on the
training split alone, they record which split they came from, they serialize
alongside the feature spec, and `transform` refuses a matrix whose columns do
not match the ones they were fitted for.

## Why the test split cannot be named here at all

`fit` refuses `test` outright rather than trusting the caller. R3 cannot load
the test split in the first place — `store.load_rollouts` has no flag for it —
so a request to fit constants on it means something has gone wrong upstream
that a silent success would hide.

## Zero-variance columns

A constant column gets a scale of 1.0, not 0.0. Dividing by its true standard
deviation is a division by zero; the resulting column of infinities then
propagates through the fit and produces a model whose coefficients are all NaN,
several steps away from the cause. A constant column carries no information
either way, so passing it through untouched loses nothing and stays debuggable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import SplitError
from .spec import FeatureMatrix, FeatureError

#: Splits normalization may be fitted on. `val` is permitted but not the
#: default: the calibrator is fitted there, and reusing it for scaling as well
#: couples two things that are cleaner apart.
FITTABLE_SPLITS: frozenset[str] = frozenset({"train", "val"})


@dataclass(frozen=True)
class Standardizer:
    """Per-column centre and scale, and the provenance to defend them."""

    names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    fitted_on: tuple[str, ...]
    n_rows: int
    #: Columns with no variance in the fitting split, passed through unscaled.
    constant_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (len(self.names) == len(self.mean) == len(self.scale)):
            raise FeatureError(
                f"standardizer has {len(self.names)} names, {len(self.mean)} "
                f"means and {len(self.scale)} scales; these must agree or the "
                f"columns are being scaled by the wrong constants"
            )

    def transform(self, matrix: FeatureMatrix) -> FeatureMatrix:
        """Apply the fitted constants. Never refits, whatever it is given."""
        if matrix.names != self.names:
            extra = sorted(set(matrix.names) - set(self.names))
            missing = sorted(set(self.names) - set(matrix.names))
            raise FeatureError(
                f"standardizer was fitted for a different feature set. "
                f"Unexpected: {extra}. Missing: {missing}. Order matters too — "
                f"a matrix with the right columns in the wrong order would be "
                f"scaled by the wrong constants and would not error."
            )
        centred = matrix.X - np.asarray(self.mean, dtype=float)
        scaled = centred / np.asarray(self.scale, dtype=float)
        return FeatureMatrix(
            X=scaled,
            names=matrix.names,
            rollout_ids=matrix.rollout_ids,
            decision_point=matrix.decision_point,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "fitted_on": list(self.fitted_on),
            "n_rows": self.n_rows,
            "constant_columns": list(self.constant_columns),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Standardizer":
        return Standardizer(
            names=tuple(data["names"]),
            mean=tuple(float(v) for v in data["mean"]),
            scale=tuple(float(v) for v in data["scale"]),
            fitted_on=tuple(data.get("fitted_on", ())),
            n_rows=int(data.get("n_rows", 0)),
            constant_columns=tuple(data.get("constant_columns", ())),
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="",
        )
        return path

    @staticmethod
    def load(path: Path | str) -> "Standardizer":
        return Standardizer.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


def fit_standardizer(matrix: FeatureMatrix,
                     rows: Sequence[Mapping[str, Any]], *,
                     on: Sequence[str] = ("train",)) -> Standardizer:
    """Fit centre and scale on the named splits only.

    `rows` must be the same rows, in the same order, that `matrix` was built
    from — the split of row *i* has to line up with row *i* of the matrix.
    Checked, because getting it wrong scales the data by constants fitted on a
    different subset and produces no error at all.
    """
    wanted = tuple(str(s) for s in on)
    if not wanted:
        raise SplitError("name at least one split to fit normalization on")

    forbidden = sorted(set(wanted) - FITTABLE_SPLITS)
    if forbidden:
        raise SplitError(
            f"refusing to fit normalization constants on {forbidden}. Fitting "
            f"anything on test — including a mean and a standard deviation — "
            f"is the leak that produces a slightly-too-good number nobody can "
            f"correct afterwards. Fittable splits: {sorted(FITTABLE_SPLITS)}."
        )

    if len(rows) != len(matrix):
        raise FeatureError(
            f"{len(rows)} rows but {len(matrix)} matrix rows; these must be "
            f"the same rows in the same order, or the split mask selects the "
            f"wrong observations and nothing errors"
        )

    mask = np.array([str(row.get("split") or "") in wanted for row in rows])
    if not mask.any():
        present = sorted({str(row.get("split") or "") for row in rows})
        raise SplitError(
            f"no rows in splits {list(wanted)}; the loaded data has {present}"
        )

    fitting = matrix.X[mask]
    mean = fitting.mean(axis=0)
    scale = fitting.std(axis=0)

    constant = scale == 0.0
    # 1.0, not the true zero. See the module docstring — dividing by zero here
    # surfaces as an all-NaN coefficient vector several steps downstream.
    scale = np.where(constant, 1.0, scale)

    return Standardizer(
        names=matrix.names,
        mean=tuple(float(v) for v in mean),
        scale=tuple(float(v) for v in scale),
        fitted_on=wanted,
        n_rows=int(mask.sum()),
        constant_columns=tuple(
            name for name, is_constant in zip(matrix.names, constant)
            if is_constant
        ),
    )
