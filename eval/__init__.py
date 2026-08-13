"""R4 — the evaluation harness.

Reads a rollout store and decides whether anything actually beat the baselines.
Owns the frozen test split, every statistic, and the report.

The dependency runs one way: this package imports `schemas`, and nothing else
from the project. It never calls a model, never runs a grader, and needs no
GPU — so a wrong number here can only come from this code, and a change in
another role's package cannot silently move a reported result.
"""

from __future__ import annotations

from .loading import (
    OPEN_SPLITS,
    Rollouts,
    StoreError,
    TestSplitError,
    TestSplitUnlock,
    load_rows,
    load_run,
    unlock_test_split,
)
from .confusion import OUTCOMES, Confusion, confusion, oracle_headroom
from .frontier import (
    MatchedComparison,
    Point,
    accuracy_at_cost,
    compare_at_matched_cost,
    frontier_dominates,
    paired_accuracy_difference,
    pareto_front,
    sweep,
)
from .heuristics import (
    ThresholdRule,
    TunedHeuristic,
    features_from_tasks,
    prompt_features,
    tune,
    tuned_frontier,
)
from .policies import Outcome, standard_baselines
from .stats import (
    Interval,
    McNemarResult,
    benjamini_hochberg,
    cluster_bootstrap,
    mcnemar_exact,
    mcnemar_sample_size,
    paired_diff_bootstrap,
    summarize,
)

__all__ = [
    "OPEN_SPLITS",
    "OUTCOMES",
    "Confusion",
    "Interval",
    "MatchedComparison",
    "McNemarResult",
    "Outcome",
    "Point",
    "Rollouts",
    "StoreError",
    "TestSplitError",
    "TestSplitUnlock",
    "ThresholdRule",
    "TunedHeuristic",
    "accuracy_at_cost",
    "benjamini_hochberg",
    "cluster_bootstrap",
    "compare_at_matched_cost",
    "confusion",
    "features_from_tasks",
    "frontier_dominates",
    "load_rows",
    "load_run",
    "mcnemar_exact",
    "mcnemar_sample_size",
    "oracle_headroom",
    "paired_accuracy_difference",
    "paired_diff_bootstrap",
    "pareto_front",
    "prompt_features",
    "standard_baselines",
    "summarize",
    "sweep",
    "tune",
    "tuned_frontier",
    "unlock_test_split",
]
