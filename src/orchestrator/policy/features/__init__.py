"""Feature builders for the two decision points.

D0 decides *which arm to try* and may read only what exists before generating.
D1 decides *whether to escalate* and may additionally read the candidate, its
extraction, and the visible-test outcome. The `heuristic_route → learned_D0`
comparison isolates learning with information held constant, and
`learned_D0 → learned_D1` isolates information with learning held constant —
which only means anything if the two feature sets are genuinely separated.

They are separated by `spec.py`, mechanically, at declaration and again at
access. See its docstring for how.
"""

from __future__ import annotations

__all__ = [
    "DECISION_POINTS",
    "Feature",
    "FeatureError",
    "FeatureMatrix",
    "FeatureSet",
    "RowView",
    "feature",
    "D0_FEATURES",
    "D1_FEATURES",
    "PROBE_FEATURES",
    "feature_set",
    "Standardizer",
    "fit_standardizer",
]

from .spec import (
    DECISION_POINTS,
    Feature,
    FeatureError,
    FeatureMatrix,
    FeatureSet,
    RowView,
    feature,
)
from .d0 import D0_FEATURES
from .d1 import D1_FEATURES, PROBE_FEATURES
from .scaling import Standardizer, fit_standardizer


def feature_set(decision_point: str, *, with_probe: bool = False) -> FeatureSet:
    """The default feature set for a decision point.

    `with_probe` adds the sibling features. They are off by default because
    they oblige the policy to pay for the extra draws they read, and a cost
    that appears without anyone deciding to spend it is how a matched-cost
    comparison stops being matched.
    """
    if decision_point == "D0":
        if with_probe:
            raise FeatureError(
                "the probe features are D1 — they read outcomes of generations "
                "that have already happened, which is not a thing D0 can do"
            )
        return D0_FEATURES
    if decision_point == "D1":
        return (D1_FEATURES + PROBE_FEATURES) if with_probe else D1_FEATURES
    raise FeatureError(
        f"unknown decision point {decision_point!r}; expected one of "
        f"{list(DECISION_POINTS)}"
    )
