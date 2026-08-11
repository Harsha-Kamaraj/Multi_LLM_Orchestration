"""Anthropic-hosted models as the two rungs.

Exists so the pipeline is runnable before any GPU is provisioned, and so the
scaffold's existing model registry stays usable. It reports `mode="serving"`
because each request is timed on its own, but note that the wall-clock includes
network round-trip and a queue this project does not control — it is a fair
serving latency for *this* deployment and is not comparable to a self-hosted
vLLM number.

**This backend cannot honour a seed.** The Messages API exposes no seed
parameter, so at temperature 0 every seed of a cell returns the same text. A
three-seed frozen ladder then pays three times for one distinct generation, and
R4's cluster bootstrap sees zero within-task variance that is not really zero.
`honors_seed = False` makes that visible to the sweep runner, which warns
instead of letting it surface as a strange confidence interval in week 4.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

from ...config import price
from .base import Backend, GenRequest, RawGeneration


def _billed(model_id: str, input_tokens: int, output_tokens: int,
            cache_read: int, cache_write: int) -> float | None:
    """USD for a call, or None when the model has no published rates here.

    Never raises. A `KeyError` from the pricing table would throw away a
    generation that has already been paid for, which is a strictly worse
    outcome than recording the row without a dollar figure.
    """
    try:
        return price(model_id, input_tokens, output_tokens, cache_read, cache_write)
    except KeyError:
        return None


class AnthropicBackend(Backend):
    """Messages API, one request per generation.

    Uses streaming and `get_final_message()` so a large `max_tokens` cannot hit
    the SDK's non-streaming timeout, while still returning a complete message
    with usage attached.
    """

    name = "anthropic"
    mode = "serving"
    honors_seed = False

    def __init__(self, small_model: str, large_model: str, *,
                 client: Any = None, max_retries: int = 3) -> None:
        super().__init__(small_model=small_model, large_model=large_model)
        self._client = client
        self.max_retries = max_retries

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            # Zero-arg constructor resolves ANTHROPIC_API_KEY or a configured
            # auth profile. Never hardcode a key.
            self._client = anthropic.Anthropic(max_retries=self.max_retries)
        return self._client

    def _kwargs(self, req: GenRequest, model_id: str) -> dict[str, Any]:
        p = req.params
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": p.max_tokens,
            "system": [
                # Stable prefix first so it caches across every task in a
                # sweep. The system block is identical for all tasks sharing a
                # template, which is most of a sweep's prefill.
                {"type": "text", "text": req.system,
                 "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": req.user}],
        }
        # The API rejects temperature and top_p together. Temperature is the
        # knob the arms actually vary, so it wins; top_p is sent only when it
        # has been moved off its default and temperature has not.
        if p.top_p != 1.0 and p.temperature == 0.0:
            kwargs["top_p"] = p.top_p
        else:
            kwargs["temperature"] = p.temperature
        if p.stop:
            kwargs["stop_sequences"] = list(p.stop)
        return kwargs

    def _one(self, req: GenRequest, batch_size: int) -> RawGeneration:
        model_id = self.model_id(req.model_role)
        t0 = time.perf_counter()
        try:
            with self.client.messages.stream(**self._kwargs(req, model_id)) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - one bad cell, not a dead sweep
            return RawGeneration.failure(
                model_id, f"{exc.__class__.__name__}: {exc}",
                mode=self.mode, batch_size=batch_size,
                wall_ms=(time.perf_counter() - t0) * 1000.0,
            )
        wall_ms = (time.perf_counter() - t0) * 1000.0

        # A safety decline returns HTTP 200 with stop_reason "refusal" and
        # possibly an empty content list, so check before indexing content.
        if message.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in message.content if b.type == "text")

        usage = message.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # Cached reads are prefill the model still processes; excluding them
        # would make prefill look like it collapsed once the cache warmed, and
        # the fitted prefill coefficient would follow it.
        prefill = int(usage.input_tokens) + int(cache_read) + int(cache_write)

        return RawGeneration(
            text=text,
            model_id=model_id,
            prefill_tokens=prefill,
            decode_tokens=int(usage.output_tokens),
            wall_ms=wall_ms,
            finish_reason=message.stop_reason or "unknown",
            mode=self.mode,
            batch_size=batch_size,
            tokens_exact=True,
            extra={
                # Billed dollars, recorded alongside the imputed GPU-second
                # cost. For a hosted backend this is the real number, and it
                # is a useful check on the imputation rather than a
                # replacement for it. `None` when the model is not in the
                # pricing registry — an unpriced model must not discard a
                # generation that already cost real money to produce.
                "billed_usd": _billed(
                    model_id, int(usage.input_tokens), int(usage.output_tokens),
                    cache_read, cache_write,
                ),
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
                "seed_honored": False,
            },
        )

    def _generate(self, requests: Sequence[GenRequest],
                  batch_size: int) -> Iterable[RawGeneration]:
        if batch_size <= 1:
            return [self._one(req, batch_size) for req in requests]
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            return list(pool.map(lambda r: self._one(r, batch_size), requests))
