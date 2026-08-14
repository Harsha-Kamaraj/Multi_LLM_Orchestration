"""Which features carry the D0 → D1 gap, and does anything else matter?

The project's headline finding is an asymmetry: `AUC_D1 ≈ 0.86` against
`AUC_D0 ≈ 0.65`. "Observing failure beats predicting it" is the sentence, and
an ablation is what turns it from a slogan into a claim — because D1 adds three
quite different things at once, and only one of them is *observing failure*:

    visible_outcome   did the visible tests pass
    code_shape        what the candidate looks like
    generation_meta   how the generation terminated

If the whole gap sits in `visible_outcome`, the finding is precisely about
verification. If `code_shape` carries most of it, the finding is something much
weaker and more ordinary — that longer, messier code is likelier to be wrong.
Those are different papers, and only an ablation separates them.

## Every number here is a paired difference

Each variant is fitted and scored, then the *difference* in AUC against the
baseline is bootstrapped over the **same resampled tasks** for both. Pairing
removes between-task difficulty, which is by far the largest source of variance
and is shared by the two variants — comparing two independently-bootstrapped
intervals instead would widen them enough to hide every real effect.

The resampling is R4's `cluster_bootstrap`, unmodified, using the same
row-index trick as `gate.py`: the matrix holds indices, and the statistic looks
up both variants' scores at those indices.

## Multiplicity is not optional here

An ablation table is a family of comparisons, and running sixteen of them at 95%
means roughly a coin-flip chance that at least one interval excludes zero by
luck alone. Reporting that one as "the group that matters" is how a null result
becomes a headline.

So the intervals are **simultaneous**: each is computed at
`1 - (1 - level) / m` for a family of `m`, which is Bonferroni on the interval
rather than on a p-value. It is conservative, it needs no second statistic, and
it keeps every number in the same shape the rest of the project reports —
a point estimate with an interval, never a bare mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from eval.stats import Interval, cluster_bootstrap

from .errors import PolicyError
from .features import FeatureSet, feature_set
from .store import RolloutData

#: Feature groups, named by hand because a `Feature` carries no group field and
#: inferring one from a name prefix would silently regroup itself the next time
#: someone renames a feature.
#:
#: The split is by *evidence*, not by convenience: everything in a group is
#: computable from the same underlying observation, so removing a group removes
#: an entire kind of knowledge rather than a few correlated columns.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "prompt_shape": (
        "prompt_chars", "prompt_words", "prompt_sentences",
        "prompt_mean_word_len",
    ),
    "visible_tests": (
        "n_visible_tests", "visible_test_chars", "visible_assert_density",
    ),
    "keywords": (
        "algorithmic_keywords", "loop_keywords", "has_code_block",
        "digit_density",
    ),
    "entrypoint": ("entrypoint_len", "entrypoint_words"),
    "visible_outcome": (
        "visible_pass_rate", "visible_all_passed", "visible_none_passed",
        "has_visible_tests", "visible_failed_count",
    ),
    "code_shape": (
        "code_chars", "code_lines", "code_parses", "code_ast_nodes",
        "code_max_depth", "defines_entrypoint", "code_has_todo",
    ),
    "generation_meta": (
        "finish_stop", "finish_length", "finish_refusal", "finish_error",
        "decode_tokens", "prefill_tokens", "extraction_was_bare", "had_error",
    ),
    "hacks": ("hack_flag_count",),
}


class AblationError(PolicyError):
    """An ablation cannot be run as asked."""


def group_of(name: str) -> str | None:
    for group, members in FEATURE_GROUPS.items():
        if name in members:
            return group
    return None


def check_groups_cover(features: FeatureSet) -> tuple[str, ...]:
    """Feature names belonging to no group.

    An ungrouped feature is invisible to every leave-one-out below, so it can
    carry an effect that the table then attributes to nothing. Surfaced rather
    than tolerated: adding a feature without adding it to a group should be a
    visible omission.
    """
    return tuple(sorted(f.name for f in features if group_of(f.name) is None))


def without_group(features: FeatureSet, group: str) -> FeatureSet:
    kept = [f for f in features if group_of(f.name) != group]
    if len(kept) == len(list(features)):
        raise AblationError(
            f"group {group!r} removes nothing from this feature set; it holds "
            f"{sorted(FEATURE_GROUPS.get(group, ()))}"
        )
    if not kept:
        raise AblationError(f"removing {group!r} would leave no features")
    return FeatureSet(kept, name=f"{features.name}-no_{group}")


def only_group(features: FeatureSet, group: str) -> FeatureSet:
    kept = [f for f in features if group_of(f.name) == group]
    if not kept:
        raise AblationError(
            f"group {group!r} contributes no feature to {features.name!r}"
        )
    return FeatureSet(kept, name=f"{features.name}-only_{group}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationDelta:
    """One variant's AUC, and its paired difference from the baseline."""

    name: str
    kind: str
    group: str
    n_features: int
    auc: float
    delta: Interval

    @property
    def significant(self) -> bool:
        """Whether the simultaneous interval supports a directional claim."""
        return self.delta.excludes_zero

    @property
    def degenerate(self) -> bool:
        """An exactly-zero delta with an exactly-zero-width interval.

        Not a finding. A group that genuinely does not help still moves the AUC
        a little under resampling, so `+0.0000 [+0.0000, +0.0000]` means the
        model never used these features at all — almost always because they are
        *constant on this data*, which is a property of the store rather than
        of the group.

        It matters because "removing this changes nothing" and "this column had
        no variation to remove" read identically in a table and mean opposite
        things. Synthetic fixtures produce the second constantly: `schemas.synth`
        writes a near-identical stub for every `code`, so every code-shape
        feature is flat and its ablation is vacuous.
        """
        return self.delta.point == 0.0 and self.delta.width == 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "group": self.group,
            "n_features": self.n_features,
            "auc": self.auc,
            "delta": self.delta.as_dict(),
            "significant": self.significant,
            "degenerate": self.degenerate,
        }


@dataclass(frozen=True)
class AblationTable:
    """A family of ablations, corrected together."""

    decision_point: str
    run_id: str
    baseline_auc: float
    baseline_features: int
    deltas: tuple[AblationDelta, ...]
    level: float
    simultaneous_level: float
    ungrouped: tuple[str, ...] = ()

    @property
    def family_size(self) -> int:
        return len(self.deltas)

    def by_impact(self) -> tuple[AblationDelta, ...]:
        """Leave-one-out variants, most damaging first."""
        loo = [d for d in self.deltas if d.kind == "leave_one_out"]
        return tuple(sorted(loo, key=lambda d: d.delta.point))

    def degenerate_groups(self) -> tuple[str, ...]:
        """Groups whose ablation was vacuous because their features were flat."""
        return tuple(sorted({d.group for d in self.deltas
                             if d.kind == "leave_one_out" and d.degenerate}))

    def carries_the_gap(self) -> AblationDelta | None:
        """The single group whose removal hurts most, if the drop is real.

        `None` when nothing is significant — which is a real answer and the one
        a wide table is most likely to produce. It means the groups are
        redundant enough that no single removal is detectable, not that they
        are all useless.
        """
        ranked = self.by_impact()
        if not ranked or not ranked[0].significant:
            return None
        return ranked[0]

    def summary(self) -> str:
        lines = [
            f"ablations at {self.decision_point} from run {self.run_id}",
            f"  baseline AUC {self.baseline_auc:.4f} on "
            f"{self.baseline_features} features",
            f"  {self.family_size} comparisons, simultaneous at "
            f"{self.simultaneous_level:.4f} (Bonferroni from {self.level:.2f})",
        ]
        for delta in self.by_impact():
            mark = "*" if delta.significant else ("?" if delta.degenerate
                                                  else " ")
            lines.append(
                f"  {mark} without {delta.group:<16} AUC {delta.auc:.4f}  "
                f"delta {delta.delta}"
            )
        best = self.carries_the_gap()
        lines.append(
            f"  -> {best.group!r} carries the most, and the drop is real"
            if best else
            "  -> no single group's removal is detectable at this family size. "
            "The groups are redundant, not useless."
        )
        vacuous = self.degenerate_groups()
        if vacuous:
            lines.append(
                f"  ? {list(vacuous)}: removal changed nothing at all, which "
                f"means these features were constant on this run rather than "
                f"unhelpful. Not a finding — check the store."
            )
        if self.ungrouped:
            lines.append(
                f"  NOTE: {len(self.ungrouped)} feature(s) belong to no group "
                f"and no leave-one-out can see them: {list(self.ungrouped)}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        best = self.carries_the_gap()
        return {
            "decision_point": self.decision_point,
            "run_id": self.run_id,
            "baseline_auc": self.baseline_auc,
            "baseline_features": self.baseline_features,
            "level": self.level,
            "simultaneous_level": self.simultaneous_level,
            "family_size": self.family_size,
            "carries_the_gap": best.group if best else None,
            "degenerate_groups": list(self.degenerate_groups()),
            "ungrouped": list(self.ungrouped),
            "deltas": [d.as_dict() for d in self.deltas],
        }


# ---------------------------------------------------------------------------
# Running them
# ---------------------------------------------------------------------------


def run_ablations(data: RolloutData, decision_point: str, *,
                  features: FeatureSet | None = None,
                  groups: Sequence[str] | None = None,
                  include_only: bool = True,
                  arm: str | None = None,
                  n_resamples: int = 1000,
                  level: float = 0.95,
                  seed: int = 0) -> AblationTable:
    """Fit the baseline and every variant, and pair each against the baseline.

    `include_only` adds the mirror comparison — that group *alone* — which is
    what separates "this group is necessary" from "this group is sufficient".
    A group can be both, either, or neither, and leave-one-out on its own
    cannot tell those apart when features are correlated.
    """
    from .gate import _fit_predict, _index_matrix, auc, cheap_arm

    features = features or feature_set(decision_point)
    target = arm or cheap_arm(data)

    rows = [r for r in data.rows if str(r["arm"]) == target]
    train_rows = [r for r in rows if str(r.get("split")) == "train"]
    eval_rows = [r for r in rows if str(r.get("split")) == "val"]
    if not train_rows or not eval_rows:
        raise AblationError(
            f"arm {target!r} needs both train and val rows; it has "
            f"{len(train_rows)} and {len(eval_rows)}"
        )

    def solved(row: Mapping[str, Any]) -> int:
        return int(data.label_for(str(row["rollout_id"])).solved)

    labels = np.array([solved(r) for r in eval_rows], dtype=int)
    matrix, _, _ = _index_matrix(eval_rows)

    base_scores, n_base = _fit_predict(train_rows, eval_rows, features, solved)
    baseline_auc = auc(base_scores, labels)

    present = sorted({group_of(f.name) for f in features} - {None})
    wanted = list(groups) if groups is not None else present
    unknown = [g for g in wanted if g not in FEATURE_GROUPS]
    if unknown:
        raise AblationError(f"unknown feature group(s): {unknown}")

    variants: list[tuple[str, str, str, FeatureSet]] = []
    for group in wanted:
        if group not in present:
            continue
        variants.append((f"no_{group}", "leave_one_out", group,
                         without_group(features, group)))
        if include_only:
            variants.append((f"only_{group}", "only", group,
                             only_group(features, group)))
    if not variants:
        raise AblationError(
            "no group in this feature set to ablate; "
            f"it contains groups {present}"
        )

    # Bonferroni on the interval: a family of m comparisons at `level` each
    # gives a much weaker simultaneous guarantee than `level`.
    simultaneous = 1.0 - (1.0 - level) / len(variants)

    deltas: list[AblationDelta] = []
    for name, kind, group, variant in variants:
        scores, n_features = _fit_predict(train_rows, eval_rows, variant, solved)
        variant_auc = auc(scores, labels)

        def delta_of(sample: np.ndarray,
                     _scores: np.ndarray = scores) -> float:
            flat = sample.ravel()
            idx = flat[~np.isnan(flat)].astype(int)
            # Same rows for both, which is the entire point of pairing.
            return auc(_scores[idx], labels[idx]) - auc(base_scores[idx],
                                                        labels[idx])

        deltas.append(AblationDelta(
            name=name,
            kind=kind,
            group=group,
            n_features=n_features,
            auc=variant_auc,
            delta=cluster_bootstrap(matrix, delta_of, n_resamples=n_resamples,
                                    level=simultaneous, seed=seed),
        ))

    return AblationTable(
        decision_point=decision_point,
        run_id=data.run_id,
        baseline_auc=baseline_auc,
        baseline_features=n_base,
        deltas=tuple(deltas),
        level=level,
        simultaneous_level=simultaneous,
        ungrouped=check_groups_cover(features),
    )
