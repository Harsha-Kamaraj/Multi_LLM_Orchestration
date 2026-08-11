"""vLLM's OpenAI-compatible server — the characterization and serving engine.

This is the *only* backend whose wall-clock is a latency measurement, and it is
one because of how it is called: a declared, fixed concurrency, with each
request timed individually. The characterization pass runs it at batch=1 and
batch=8, fits `latency ~ prefill + decode` per model, and those coefficients
are what every imputed latency in the project is built from.

Requests go out over `urllib` from the standard library rather than through the
`openai` package. The surface used here is one POST and three fields of the
response; a dependency for that would be a dependency a sweep host has to have
installed, for no benefit.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence

from .base import Backend, GenRequest, RawGeneration

# Retried: the server is still loading weights, is briefly overloaded, or the
# connection dropped. Not retried: a 400, which means the request itself is
# wrong and will be wrong again.
_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class VLLMOpenAIBackend(Backend):
    """Chat completions against an OpenAI-compatible server.

    `concurrency` passed to `generate` is the declared in-flight request count
    and is recorded as `batch_size` on every row, because a latency number
    without the concurrency it was measured at is not a number.
    """

    name = "vllm_openai"
    mode = "serving"

    def __init__(self, small_model: str, large_model: str, *,
                 base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY",
                 timeout_s: float = 600.0,
                 max_retries: int = 3,
                 retry_backoff_s: float = 2.0) -> None:
        super().__init__(small_model=small_model, large_model=large_model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    # -- transport -----------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code not in _RETRY_STATUSES:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = RuntimeError(f"{exc.__class__.__name__}: {exc}")
            if attempt < self.max_retries:
                # Linear rather than exponential: the common case is a server
                # still warming up, where a fixed short wait recovers faster
                # than doubling, and the retry budget is small enough that
                # exponential buys nothing.
                time.sleep(self.retry_backoff_s * (attempt + 1))
        raise last or RuntimeError("request failed with no recorded cause")

    # -- generation ----------------------------------------------------------

    def _payload(self, req: GenRequest, model_id: str) -> dict[str, Any]:
        p = req.params
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "temperature": p.temperature,
            "top_p": p.top_p,
            "max_tokens": p.max_tokens,
            "n": 1,
            "stream": False,
        }
        if p.stop:
            payload["stop"] = list(p.stop)
        if p.seeded:
            payload["seed"] = req.seed
        # vLLM accepts sampler knobs the OpenAI schema has no field for through
        # this passthrough, so top_k and min_p survive the HTTP hop instead of
        # being silently dropped — a dropped sampler is a params_hash that no
        # longer describes what ran.
        extra = {}
        if p.top_k is not None:
            extra["top_k"] = p.top_k
        if p.min_p is not None:
            extra["min_p"] = p.min_p
        if extra:
            payload.update(extra)
        return payload

    def _one(self, req: GenRequest, batch_size: int) -> RawGeneration:
        model_id = self.model_id(req.model_role)
        t0 = time.perf_counter()
        try:
            data = self._post("/chat/completions", self._payload(req, model_id))
        except Exception as exc:  # noqa: BLE001 - one bad cell, not a dead sweep
            return RawGeneration.failure(
                model_id, f"{exc.__class__.__name__}: {exc}",
                mode=self.mode, batch_size=batch_size,
                wall_ms=(time.perf_counter() - t0) * 1000.0,
            )
        wall_ms = (time.perf_counter() - t0) * 1000.0

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return RawGeneration(
            text=message.get("content") or "",
            model_id=data.get("model") or model_id,
            prefill_tokens=int(usage.get("prompt_tokens") or 0),
            decode_tokens=int(usage.get("completion_tokens") or 0),
            wall_ms=wall_ms,
            finish_reason=choice.get("finish_reason") or "unknown",
            mode=self.mode,
            batch_size=batch_size,
            # Reported by the server's own tokenizer, which is the serving
            # tokenizer by definition.
            tokens_exact=bool(usage),
        )

    def _generate(self, requests: Sequence[GenRequest],
                  batch_size: int) -> Iterable[RawGeneration]:
        if batch_size <= 1:
            return [self._one(req, batch_size) for req in requests]
        # `map` preserves input order, so results stay positionally aligned
        # with requests even though they complete out of order.
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            return list(pool.map(lambda r: self._one(r, batch_size), requests))

    # -- health --------------------------------------------------------------

    def ping(self) -> list[str]:
        """Model ids the server is actually serving.

        Worth calling before a characterization pass: a server serving
        different weights than the sweep used produces coefficients that are
        precise, plausible, and about the wrong model.
        """
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("id", "") for m in data.get("data", [])]
