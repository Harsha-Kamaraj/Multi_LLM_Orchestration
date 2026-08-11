"""vLLM offline batch generation — the sweep engine.

`LLM.generate` is handed the whole batch at once and tuned purely for
throughput. Continuous batching keeps the GPU saturated, which is the right
objective for a sweep and the wrong one for measuring anything about time.

**Wall-clock from this backend is not latency.** With 256 requests in flight,
any single request's elapsed time is dominated by queue depth. It is recorded
because a sudden change in it is a useful signal that something is wrong with
the serving stack, and every row it writes carries `mode="sweep"`, which the
characterization pass filters on. Latency coefficients come from the
OpenAI-server backend at declared concurrency, never from here.

Both rungs are served by **one engine at a time**. Loading two models
concurrently is how a sweep runs out of VRAM four hours in; the sweep runner
groups requests by arm so each rung is generated in one pass.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

from ..errors import BackendError
from .base import Backend, GenRequest, RawGeneration


class VLLMOfflineBackend(Backend):
    """Batched generation through vLLM's offline `LLM` interface.

    The engine is constructed lazily on first use and swapped when a request
    needs the other rung, so only one model's weights are resident at a time.
    """

    name = "vllm_offline"
    mode = "sweep"

    def __init__(self, small_model: str, large_model: str, *,
                 dtype: str = "auto",
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: int | None = None,
                 tensor_parallel_size: int = 1,
                 enforce_eager: bool = False,
                 trust_remote_code: bool = False,
                 engine_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__(small_model=small_model, large_model=large_model)
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.trust_remote_code = trust_remote_code
        self.engine_kwargs = dict(engine_kwargs or {})
        self._llm: Any = None
        self._loaded_model: str | None = None

    # -- engine lifecycle ----------------------------------------------------

    def _engine(self, model_id: str) -> Any:
        """Return an engine serving `model_id`, loading or swapping as needed."""
        if self._loaded_model == model_id and self._llm is not None:
            return self._llm
        self.close()

        from vllm import LLM  # imported here: constructing it allocates VRAM

        kwargs: dict[str, Any] = {
            "model": model_id,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": self.enforce_eager,
            "trust_remote_code": self.trust_remote_code,
            **self.engine_kwargs,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**kwargs)
        self._loaded_model = model_id
        return self._llm

    def close(self) -> None:
        """Drop the engine and free VRAM.

        Deleting the handle is not enough — vLLM holds the KV cache through the
        distributed executor, and without an explicit collect the next model
        loads alongside the old one's allocation and OOMs.
        """
        if self._llm is None:
            self._loaded_model = None
            return
        try:
            import gc

            try:
                from vllm.distributed.parallel_state import (  # type: ignore
                    destroy_model_parallel,
                )

                destroy_model_parallel()
            except Exception:  # noqa: BLE001 - best effort across vLLM versions
                pass
            del self._llm
            gc.collect()
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._llm = None
            self._loaded_model = None

    # -- generation ----------------------------------------------------------

    def _sampling_params(self, req: GenRequest) -> Any:
        from vllm import SamplingParams

        p = req.params
        kwargs: dict[str, Any] = {
            "temperature": p.temperature,
            "top_p": p.top_p,
            "max_tokens": p.max_tokens,
            "n": 1,
        }
        if p.top_k is not None:
            kwargs["top_k"] = p.top_k
        if p.min_p is not None:
            kwargs["min_p"] = p.min_p
        if p.stop:
            kwargs["stop"] = list(p.stop)
        if p.seeded:
            # Recorded as row identity regardless. vLLM's batching means a seed
            # does not make greedy decoding bitwise reproducible across batch
            # sizes, so this pins what can be pinned and no more.
            kwargs["seed"] = req.seed
        return SamplingParams(**kwargs)

    def _prompts(self, requests: Sequence[GenRequest], model_id: str) -> list[str]:
        """Apply the model's chat template to each request.

        The template is the model's own, taken from the serving tokenizer, so
        the prompt string the engine tokenizes is exactly the one a served
        request would produce. Hand-rolling the chat format here is how prefill
        token counts drift away from what the server would report.
        """
        tokenizer = self._engine(model_id).get_tokenizer()
        prompts = []
        for req in requests:
            messages = [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001 - base models have no chat template
                text = f"{req.system}\n\n{req.user}\n"
            prompts.append(text)
        return prompts

    def _generate(self, requests: Sequence[GenRequest],
                  batch_size: int) -> Iterable[RawGeneration]:
        models = {self.model_id(r.model_role) for r in requests}
        if len(models) > 1:
            raise BackendError(
                f"offline batch spans {len(models)} models ({sorted(models)}); "
                f"group requests by arm so one engine serves one batch — "
                f"swapping weights mid-batch defeats continuous batching and "
                f"risks running both models resident at once"
            )
        model_id = models.pop()

        llm = self._engine(model_id)
        prompts = self._prompts(requests, model_id)
        sampling = [self._sampling_params(r) for r in requests]

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        if len(outputs) != len(requests):
            raise BackendError(
                f"vLLM returned {len(outputs)} outputs for {len(requests)} prompts"
            )

        for out in outputs:
            completion = out.outputs[0]
            yield RawGeneration(
                text=completion.text,
                model_id=model_id,
                # Both counts come from the engine's own tokenizer, which is
                # the serving tokenizer by construction.
                prefill_tokens=len(out.prompt_token_ids or ()),
                decode_tokens=len(completion.token_ids or ()),
                # The whole batch's elapsed time, on every row in it. Divided
                # per request it would look like a per-request measurement,
                # which is precisely the misreading to avoid.
                wall_ms=wall_ms,
                finish_reason=completion.finish_reason or "unknown",
                mode=self.mode,
                batch_size=batch_size,
                tokens_exact=True,
                extra={"cumulative_logprob": getattr(completion, "cumulative_logprob", None)},
            )
