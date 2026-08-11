"""Arm definitions — the unit the policy chooses between.

An arm is a *generation configuration*: which rung of the model ladder to use,
with which sampling parameters, at which step of the repair ladder. It is not a
model. `direct_small` and `probe_small` run the same weights and are different
arms, because they differ in what they cost and in what they produce.

Arms name a **role** (`small` / `large`) rather than a model id. The backend
resolves the role, so re-running the whole sweep against a different pair of
models is a backend configuration change and not an edit to this file. The
resolved `model_id` is recorded on every row, so nothing is lost by the
indirection.

`ladder_step` and `requires_parent` exist from day one even though the repair
ladder is a Phase 2 deliverable. The rollout schema carries `parent_rollout_id`
and `ladder_step`, and a field that is only populated later is a field that gets
populated wrongly — the plumbing is cheaper to build now than to retrofit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .params import GREEDY, SAMPLED, GenParams

ModelRole = Literal["small", "large"]


@dataclass(frozen=True)
class Arm:
    """One generation configuration, resolvable by any backend."""

    name: str
    model_role: ModelRole
    params: GenParams
    # Position in the repair ladder. Step 0 arms generate from the task alone;
    # step >0 arms are seeded from a parent generation and its test feedback.
    ladder_step: int = 0
    requires_parent: bool = False
    description: str = ""

    @property
    def params_hash(self) -> str:
        return self.params.params_hash

    def __post_init__(self) -> None:
        # A step >0 arm with no parent would silently generate from scratch and
        # be recorded as a repair, which would make the repair ladder look
        # ineffective for reasons that have nothing to do with repair.
        if self.ladder_step > 0 and not self.requires_parent:
            raise ValueError(
                f"arm {self.name!r} is at ladder step {self.ladder_step} but does "
                f"not require a parent; a repair with no parent is not a repair"
            )
        if self.requires_parent and self.ladder_step == 0:
            raise ValueError(
                f"arm {self.name!r} requires a parent but sits at ladder step 0"
            )


ARMS: dict[str, Arm] = {
    "direct_small": Arm(
        name="direct_small",
        model_role="small",
        params=GREEDY,
        description="Small model, greedy, visible tests shown. The cost floor.",
    ),
    "direct_large": Arm(
        name="direct_large",
        model_role="large",
        params=GREEDY,
        description="Large model, greedy, visible tests shown. The accuracy ceiling.",
    ),
    # Ablation pair. The gap against the arms above is how much of each model's
    # accuracy comes from being shown the visible tests rather than from the
    # model — which is worth knowing before attributing any of it to routing.
    "direct_small_notests": Arm(
        name="direct_small_notests",
        model_role="small",
        params=GREEDY.evolve(template_id="direct_notests_v1"),
        description="Small model, greedy, no visible tests. Ablation.",
    ),
    "direct_large_notests": Arm(
        name="direct_large_notests",
        model_role="large",
        params=GREEDY.evolve(template_id="direct_notests_v1"),
        description="Large model, greedy, no visible tests. Ablation.",
    ),
    # Sampled small-model draws. One arm serves both `best_of_n_small` and the
    # self-consistency probe: they differ in how R3/R4 *use* the k draws, not
    # in what is generated, so generating them twice would be waste.
    "probe_small": Arm(
        name="probe_small",
        model_role="small",
        params=SAMPLED,
        description="Small model, sampled. Feeds best-of-n and the self-consistency probe.",
    ),
    # Phase 2. Seeded from a parent generation plus its visible-test feedback.
    "repair_small": Arm(
        name="repair_small",
        model_role="small",
        params=GREEDY.evolve(template_id="repair_v1"),
        ladder_step=1,
        requires_parent=True,
        description="One round of small-model repair on a failed attempt.",
    ),
    "repair_large": Arm(
        name="repair_large",
        model_role="large",
        params=GREEDY.evolve(template_id="repair_v1"),
        ladder_step=1,
        requires_parent=True,
        description="One round of large-model repair on a failed attempt.",
    ),
}

# The Phase 1 frozen ladder: two arms, three seeds each, six generations per
# task. Anything beyond this is opt-in per sweep — a sweep that quietly grew an
# arm is a sweep whose cost accounting no longer matches its manifest.
FROZEN_LADDER: tuple[str, ...] = ("direct_small", "direct_large")

# Phase 0's pilot seeds. Three is the smallest number that lets a per-task
# variance estimate exist at all, which R4 needs for the cluster bootstrap.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)


def get_arm(name: str) -> Arm:
    """Look up an arm, failing loudly on a typo.

    No default. Falling back would generate under a different configuration
    than the one recorded on the row.
    """
    try:
        return ARMS[name]
    except KeyError:
        raise KeyError(
            f"unknown arm {name!r}; known arms: {sorted(ARMS)}"
        ) from None


def resolve_arms(names: "list[str] | tuple[str, ...] | None") -> tuple[Arm, ...]:
    """Resolve arm names to `Arm` objects, defaulting to the frozen ladder."""
    selected = tuple(names) if names else FROZEN_LADDER
    return tuple(get_arm(n) for n in selected)
