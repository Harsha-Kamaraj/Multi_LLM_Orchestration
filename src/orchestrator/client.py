"""Thin Anthropic wrapper that records cost and latency for every call.

Nothing else in the codebase talks to the API directly, so this is the single
place where token usage becomes dollars and wall-clock becomes latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic

from .config import LARGE_MODEL, RunConfig, price
from .types import Usage


@dataclass
class Call:
    text: str
    usage: Usage
    cost_usd: float
    latency_s: float
    stop_reason: str | None = None


class MeteredClient:
    """Wraps the Anthropic SDK and meters each request.

    Uses streaming with `get_final_message()` so large `max_tokens` values
    never hit the SDK's HTTP timeout, while still returning a complete message.
    """

    def __init__(self, cfg: RunConfig | None = None,
                 client: anthropic.Anthropic | None = None) -> None:
        self.cfg = cfg or RunConfig()
        # Zero-arg constructor resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile — don't hardcode a key.
        self.client = client or anthropic.Anthropic()
        self.total_cost_usd = 0.0
        self.total_calls = 0

    def complete(self, model: str, system: str, user: str,
                 max_tokens: int | None = None,
                 effort: str | None = None) -> Call:
        max_tokens = max_tokens or self.cfg.max_tokens
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                # Stable prefix first so it caches across every task in a sweep.
                {"type": "text", "text": system,
                 "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user}],
        }

        # Effort and adaptive thinking exist on the frontier models only;
        # Haiku 4.5 rejects `effort` outright.
        if model == LARGE_MODEL or model.startswith("claude-opus-5"):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort or self.cfg.large_effort}

        t0 = time.perf_counter()
        with self.client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        latency = time.perf_counter() - t0

        # A safety decline returns HTTP 200 with stop_reason "refusal" and
        # possibly an empty content list — check before indexing content.
        if message.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in message.content if b.type == "text")

        u = message.usage
        usage = Usage(
            model=model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        cost = price(
            model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        )
        self.total_cost_usd += cost
        self.total_calls += 1
        return Call(text=text, usage=usage, cost_usd=cost,
                    latency_s=latency, stop_reason=message.stop_reason)
