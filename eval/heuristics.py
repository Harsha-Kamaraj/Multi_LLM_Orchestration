"""`heuristic_route` — the ablation of learning itself.

This baseline answers the question an interviewer asks first: *couldn't you have
just used prompt length?* Without it the answer is a shrug; with it the answer
is a number.

It sits directly below `learned_D0` on the capacity ladder, holding the
information set constant and varying only whether the rule was learned:

    random_route     0 parameters,      no information
    heuristic_route  1-2 thresholds,    prompt only
    learned_D0       dozens,            prompt only
    learned_D1       dozens,            prompt + code + visible tests

So `heuristic_route -> learned_D0` isolates **learning**, and
`learned_D0 -> learned_D1` isolates **information**.

Two rules make this an honest baseline rather than a strawman, and both are
enforced here rather than trusted:

**It is tuned as hard as the policy.** Thresholds are fit on the validation
split — the same split the policy uses — and swept across λ to produce a
frontier. Comparing a tuned curve against an untuned point is not a comparison.

**It stays a heuristic.** Hand-specified rules with one or two free parameters.
The moment you fit a model over prompt features you have rebuilt `learned_D0`,
and the ablation collapses to noise. `MAX_FREE_PARAMETERS` is checked.

Owned by R4 by design: the person whose policy is being measured must not
control its competition.

## Where the features come from

**The rollout row carries no prompt.** It carries `text` and `code` — what the
model produced — and the `task_id` that produced them. Every prompt-only signal
here is therefore uncomputable from the rollout store alone, and comes from R2's
task manifest joined by `task_id`.

That join is the whole reason this file takes `Features` as an argument rather
than reading a store: it keeps the two sources visibly separate, so a signal
derived from a model's *output* can never be mistaken for one available *before
generating*. A D0 baseline that accidentally reads `code` is not a D0 baseline,
and the mistake would look like the heuristic performing well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .loading import Rollouts
from .policies import Outcome, from_decisions

# A "heuristic" with more knobs than this is a model wearing a disguise.
MAX_FREE_PARAMETERS = 2

Features = Mapping[str, Mapping[str, float]]  # task_id -> name -> value

_ALGORITHMIC = re.compile(
    r"\b(graph|dynamic|recursi\w*|optimi[sz]\w*|permutation|combinat\w*|"
    r"backtrack\w*|matrix|modulo|prime|binary search|dijkstra|knapsack)\b",
    re.IGNORECASE,
)
_NESTING = re.compile(r"\b(for|while|if)\b")


def prompt_features(task: Mapping[str, object]) -> dict[str, float]:
    """Prompt-only signals for one task.

    Every feature here is computable from the prompt and the *visible* tests
    alone — the information a router has before generating anything. Nothing
    reads the hidden tests, and nothing reads a dataset-provided difficulty:
    a label derived from model pass rates is leakage in disguise, however it
    is packaged.
    """
    prompt = str(task.get("prompt", ""))
    visible = str(task.get("visible_tests", ""))
    return {
        "prompt_chars": float(len(prompt)),
        "prompt_words": float(len(prompt.split())),
        "prompt_lines": float(prompt.count("\n") + 1),
        "n_visible_tests": float(visible.count("assert")),
        "control_keywords": float(len(_NESTING.findall(prompt))),
        "algorithmic_terms": float(len(_ALGORITHMIC.findall(prompt))),
    }


def features_from_tasks(tasks: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """Build the feature table for a corpus of `Task` records."""
    return {str(t["task_id"]): prompt_features(t) for t in tasks}


@dataclass(frozen=True)
class ThresholdRule:
    """Escalate when one prompt signal crosses a threshold.

    Exactly two free parameters: which side of the threshold escalates, and
    where the threshold sits. The feature choice is a discrete family member,
    reported alongside the result rather than hidden inside it.
    """

    feature: str
    threshold: float
    escalate_above: bool
    small: str = "small"
    large: str = "large"

    @property
    def n_free_parameters(self) -> int:
        return 2  # threshold, direction

    def __str__(self) -> str:
        op = ">" if self.escalate_above else "<="
        return f"escalate if {self.feature} {op} {self.threshold:.4g}"

    def actions(self, features: Features, tasks: Sequence[str]) -> dict[str, str]:
        """Decide an arm for every task. Missing features are an error.

        Defaulting a missing feature would attribute the default's outcome to
        the heuristic, which is how a baseline quietly becomes a different
        baseline.
        """
        out: dict[str, str] = {}
        for task in tasks:
            row = features.get(task)
            if row is None or self.feature not in row:
                raise KeyError(
                    f"no {self.feature!r} feature for task {task!r}; a heuristic "
                    f"must decide for every task rather than default silently"
                )
            value = row[self.feature]
            escalate = value > self.threshold if self.escalate_above else value <= self.threshold
            out[task] = self.large if escalate else self.small
        return out

    def policy(self, features: Features) -> Callable[[Rollouts], Outcome]:
        def build(store: Rollouts) -> Outcome:
            actions = self.actions(features, store.ordered_tasks)
            return from_decisions(
                actions, f"heuristic_route[{self}]",
                small=self.small, large=self.large,
            )(store)

        return build


@dataclass(frozen=True)
class TunedHeuristic:
    """A rule, the λ it was tuned for, and where it was tuned.

    `fit_split` is carried so a report can never claim a held-out number for a
    rule tuned on the split it is reported against.
    """

    rule: ThresholdRule
    lam: float
    fit_split: tuple[str, ...]
    fit_utility: float
    family: str = "threshold"


def candidate_thresholds(values: np.ndarray, *, n: int = 32) -> np.ndarray:
    """Quantile-spaced cut points.

    Quantiles rather than a uniform grid: a uniform grid over a skewed feature
    wastes most of its candidates in an empty tail, and the tuned heuristic
    would lose to the policy for a reason that has nothing to do with learning.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(0)
    qs = np.linspace(0, 100, min(n, max(2, finite.size)))
    return np.unique(np.percentile(finite, qs))


def tune(
    features: Features,
    store: Rollouts,
    lam: float,
    *,
    feature_names: Sequence[str] | None = None,
    small: str = "small",
    large: str = "large",
) -> TunedHeuristic:
    """Fit the best threshold rule at cost weight `lam`.

    Searches every (feature, threshold, direction) triple and keeps the one
    maximizing `accuracy - lam * cost` on the split it is given. The caller is
    responsible for giving it the **validation** split; `fit_split` records
    what it actually saw so a later report cannot misattribute it.
    """
    tasks = store.ordered_tasks
    names = list(feature_names or _available_features(features, tasks))
    if not names:
        raise ValueError("no prompt features available to tune on")

    best: TunedHeuristic | None = None
    for name in names:
        values = np.array(
            [float(features[t][name]) for t in tasks if name in features.get(t, {})],
            dtype=float,
        )
        if values.size != len(tasks):
            continue
        for threshold in candidate_thresholds(values):
            for above in (True, False):
                rule = ThresholdRule(name, float(threshold), above, small, large)
                utility = float(
                    rule.policy(features)(store).per_task().utility(lam).mean()
                )
                if best is None or utility > best.fit_utility:
                    best = TunedHeuristic(
                        rule=rule, lam=lam, fit_split=store.splits,
                        fit_utility=utility,
                    )

    if best is None:
        raise ValueError("threshold search produced no candidate rule")
    if best.rule.n_free_parameters > MAX_FREE_PARAMETERS:
        raise ValueError(
            f"rule has {best.rule.n_free_parameters} free parameters, above the "
            f"{MAX_FREE_PARAMETERS} that keeps this a heuristic rather than a model"
        )
    return best


def _available_features(features: Features, tasks: Sequence[str]) -> list[str]:
    """Feature names present for *every* task. A feature covering only part of
    the corpus would make the heuristic's task set differ from the policy's."""
    if not tasks:
        return []
    common: set[str] | None = None
    for task in tasks:
        keys = set(features.get(task, {}))
        common = keys if common is None else (common & keys)
    return sorted(common or set())


def tuned_frontier(
    features: Features,
    fit_store: Rollouts,
    eval_store: Rollouts,
    lams: Sequence[float],
    *,
    small: str = "small",
    large: str = "large",
) -> list[tuple[float, TunedHeuristic, Outcome]]:
    """Tune on `fit_store`, evaluate on `eval_store`, once per λ.

    This is the shape the comparison requires: a **frontier**, not a point. A
    single tuned threshold compared against a policy swept across λ is a dot
    against a curve, and the curve wins for free.

    Passing the same store as both arguments is legitimate only for diagnosing
    the optimism gap — never for a reported number.
    """
    out: list[tuple[float, TunedHeuristic, Outcome]] = []
    for lam in lams:
        best = tune(features, fit_store, lam, small=small, large=large)
        outcome = best.rule.policy(features)(eval_store)
        out.append((lam, best, outcome))
    return out


def synth_features(rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """Prompt features for the synthetic fixture.

    The generator plants a prompt-only proxy under `extra._synth_x_d0`, so the
    heuristic has a signal of *known, deliberately weak* strength to find. This
    is what makes the fixture able to test the tuning code: a heuristic that
    cannot beat `always_small` on data with a planted D0 signal has a bug.
    """
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        extra = row.get("extra") or {}
        if "_synth_x_d0" in extra:  # type: ignore[operator]
            out[str(row["task_id"])] = {
                # Negated: the planted proxy scores *easiness*, and the rule
                # escalates on hard tasks. Keeping the sign explicit here means
                # the tuner is not silently correcting an inverted feature.
                "difficulty_proxy": -float(extra["_synth_x_d0"]),  # type: ignore[index]
            }
    return out
