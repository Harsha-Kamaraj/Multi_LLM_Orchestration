"""Model registry, pricing, and the knobs that define the reward.

Pricing is USD per million tokens, first-party Anthropic API rates.
Keep this table in sync with the published pricing page — the whole
cost side of the reward is derived from it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    # Cache reads bill at ~0.1x input, writes at ~1.25x (5-minute TTL).
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


MODELS: dict[str, ModelSpec] = {
    "claude-opus-5": ModelSpec("claude-opus-5", 5.00, 25.00),
    "claude-sonnet-5": ModelSpec("claude-sonnet-5", 3.00, 15.00),
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", 1.00, 5.00),
}

# The two rungs the policy chooses between. Overridable by env so a sweep can
# re-run the whole pipeline against a different pair without code edits.
SMALL_MODEL = os.environ.get("ORCH_SMALL_MODEL", "claude-haiku-4-5")
LARGE_MODEL = os.environ.get("ORCH_LARGE_MODEL", "claude-opus-5")


def price(model: str, input_tokens: int, output_tokens: int,
          cache_read: int = 0, cache_write: int = 0) -> float:
    """Cost in USD for one call's token usage."""
    spec = MODELS.get(model)
    if spec is None:
        raise KeyError(
            f"unknown model {model!r}; add it to MODELS with its published rates"
        )
    m = 1_000_000
    return (
        input_tokens * spec.input_per_mtok / m
        + output_tokens * spec.output_per_mtok / m
        + cache_read * spec.input_per_mtok * spec.cache_read_multiplier / m
        + cache_write * spec.input_per_mtok * spec.cache_write_multiplier / m
    )


@dataclass(frozen=True)
class RewardConfig:
    """Reward = accuracy - lambda_cost * (cost/ref) - lambda_latency * (lat/ref).

    The reference values normalize cost and latency into roughly [0, 1] for a
    typical call, so the lambdas read directly as "how much accuracy is one
    reference-unit of spend worth". Defaults are deliberately conservative:
    accuracy dominates, and the penalties only decide close calls.
    """

    lambda_cost: float = 0.30
    lambda_latency: float = 0.10
    cost_ref_usd: float = 0.02
    latency_ref_s: float = 20.0
    # Use strict solved/not-solved rather than partial test credit.
    binary_accuracy: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Everything that shapes a rollout sweep."""

    small_model: str = SMALL_MODEL
    large_model: str = LARGE_MODEL
    max_tokens: int = 16000
    # Effort applies to the large model only (Haiku has no effort parameter).
    large_effort: str = "medium"
    grader_timeout_s: float = 60.0
    grader_backend: str = os.environ.get("ORCH_GRADER", "subprocess")
    decompose_max_subtasks: int = 4
