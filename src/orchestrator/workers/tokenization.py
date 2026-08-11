"""Token counting, and honesty about where the count came from.

`prefill_tokens` must come from the **serving tokenizer**, not from an
estimate. Every cost number in the project is built on it: `gpu_seconds` is
linear in prefill and decode tokens, USD is linear in `gpu_seconds`, and the
cost-accuracy frontier is drawn against USD. An estimate at the bottom of that
stack propagates all the way to the headline result, invisibly.

Backends therefore report token counts from the serving layer itself wherever
one exists — vLLM returns them per request, and both HTTP APIs return usage. A
tokenizer here is used for the cases the serving layer cannot cover: sizing a
prompt before sending it, and counting for a backend whose response omits usage.

Every tokenizer declares whether it is `exact`. An approximate count sets
`tokens_exact=False` on the row, and the characterization pass refuses to fit
on rows where it is false. An approximation that silently entered the fit would
produce coefficients that look fine and are wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Rough characters-per-token for code in the Qwen/Llama BPE family. Only used
# by the approximate fallback, whose output never reaches a cost fit.
_CHARS_PER_TOKEN = 3.6


class Tokenizer(Protocol):
    """Counts tokens the way the serving stack will count them."""

    exact: bool

    def count(self, text: str) -> int: ...


@dataclass
class ApproxTokenizer:
    """Character-ratio fallback. Never exact, and says so.

    Exists so a sweep can size prompts on a machine with no model weights
    present. Any row counted this way is marked and excluded from cost fitting.
    """

    chars_per_token: float = _CHARS_PER_TOKEN
    exact: bool = False
    name: str = "approx"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, round(len(text) / self.chars_per_token))


@dataclass
class HFTokenizer:
    """Wraps a Hugging Face fast tokenizer — the actual serving tokenizer.

    `add_special_tokens=False` matches how a chat template is applied at serve
    time: the template adds its own special tokens, and counting them twice
    inflates prefill by a handful of tokens per request. Small per request,
    systematic across a sweep, and it biases the fitted prefill coefficient.
    """

    tokenizer: object
    name: str = ""
    exact: bool = True

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))


_CACHE: dict[str, Tokenizer] = {}


def get_tokenizer(model_id: str, *, allow_approx: bool = True) -> Tokenizer:
    """Load the serving tokenizer for a model, cached per process.

    Loading is deferred and cached because a sweep counts tokens for every
    request and `transformers` is a heavy import that most of this package
    does not need.

    Set `allow_approx=False` anywhere an exact count is required — the
    characterization pass does — so a missing tokenizer fails loudly instead of
    quietly degrading the coefficients everything else is built on.
    """
    if model_id in _CACHE:
        tok = _CACHE[model_id]
        if tok.exact or allow_approx:
            return tok

    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        hf = AutoTokenizer.from_pretrained(model_id)
        tok = HFTokenizer(tokenizer=hf, name=model_id)
        _CACHE[model_id] = tok
        return tok
    except Exception as exc:  # noqa: BLE001 - any failure here means "no tokenizer"
        if not allow_approx:
            raise RuntimeError(
                f"no exact tokenizer available for {model_id!r} and an exact count "
                f"was required; install `transformers` and make the weights "
                f"reachable, or characterize on a host that can load them "
                f"({exc.__class__.__name__}: {exc})"
            ) from exc

    tok = ApproxTokenizer()
    _CACHE[model_id] = tok
    return tok


def clear_cache() -> None:
    """Drop cached tokenizers. Used by tests; harmless in production."""
    _CACHE.clear()
