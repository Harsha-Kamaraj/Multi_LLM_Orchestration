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
