"""Backend registry.

Backends are registered by name and constructed lazily. The lazy part matters:
`vllm` imports CUDA and allocates GPU memory at construction, and importing
this package must stay free on a laptop that has neither.

Selecting a backend is a configuration decision, made once per sweep and
recorded on every row it produces.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from ..errors import BackendUnavailable
from .base import Backend, GenRequest, RawGeneration

__all__ = [
    "Backend", "GenRequest", "RawGeneration",
    "register", "get_backend", "available", "default_backend_name",
]

_REGISTRY: dict[str, Callable[..., Backend]] = {}


def register(name: str, factory: Callable[..., Backend]) -> None:
    """Register a backend factory under a name."""
    _REGISTRY[name] = factory


def available() -> list[str]:
    """Registered backend names, sorted."""
    return sorted(_REGISTRY)


def default_backend_name() -> str:
    """Which backend to use when a sweep does not name one.

    `mock` is the default on purpose. A sweep that silently reaches for real
    weights, or bills a real API, because someone forgot a flag is a worse
    failure than one that obviously produced simulated rows.
    """
    return os.environ.get("ORCH_BACKEND", "mock")


def _load_mock(**kwargs: Any) -> Backend:
    from .mock import MockBackend

    return MockBackend(**kwargs)


def _load_vllm_offline(**kwargs: Any) -> Backend:
    from .vllm_offline import VLLMOfflineBackend

    return VLLMOfflineBackend(**kwargs)


def _load_vllm_openai(**kwargs: Any) -> Backend:
    from .vllm_openai import VLLMOpenAIBackend

    return VLLMOpenAIBackend(**kwargs)


def _load_anthropic(**kwargs: Any) -> Backend:
    from .anthropic_api import AnthropicBackend

    return AnthropicBackend(**kwargs)


register("mock", _load_mock)
register("vllm_offline", _load_vllm_offline)
register("vllm_openai", _load_vllm_openai)
register("anthropic", _load_anthropic)


def get_backend(name: str | None = None, **kwargs: Any) -> Backend:
    """Construct a backend by name.

    An `ImportError` from the factory is re-raised as `BackendUnavailable`,
    which separates "this backend's runtime is not installed" from "this
    backend failed to generate". Retrying helps with one and never with the
    other.
    """
    name = name or default_backend_name()
    factory = _REGISTRY.get(name)
    if factory is None:
        raise BackendUnavailable(
            f"unknown backend {name!r}; registered: {available()}"
        )
    try:
        return factory(**kwargs)
    except ImportError as exc:
        raise BackendUnavailable(
            f"backend {name!r} needs a dependency that is not installed: {exc}"
        ) from exc
