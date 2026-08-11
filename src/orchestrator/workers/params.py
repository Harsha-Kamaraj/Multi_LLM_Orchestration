"""Sampling parameters and the hash that makes them part of a row's identity.

`params_hash` is the mechanism that stops a mid-sweep parameter or prompt tweak
from silently poisoning the rollout store. It is checked on resume: a cell
computed under a different `params_hash` is never mistaken for a hit, so the
worst case of an accidental edit is recomputation, never contamination.

**Everything that changes what the model emits belongs in the hash.** The role
doc names temperature, top_p, max_tokens, and the prompt template as the
minimum; this module also folds in top_k, min_p, stop sequences, and the
presence of a seed, because each of those changes the output distribution and
would otherwise let a changed setting reuse a stale cell.

Two things are deliberately *excluded*:

* **`model_id`** — the arm determines the model, and the arm is already part of
  the resume key `(task_id, arm, seed, params_hash)`. Folding it in twice buys
  nothing.
* **`template_id`** — the template's *text* is hashed, not its label. Renaming
  a template without editing it produces identical generations and should
  reuse the cells; editing one without renaming it must not, and does not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from .prompts import PromptTemplate, get_template

# Bumped only when the *composition* of the hash changes — a new field folded
# in, or a field removed. Included in the hashed payload so a change to the
# recipe cannot produce a collision with hashes computed under the old one.
PARAMS_HASH_VERSION = 1


@dataclass(frozen=True)
class GenParams:
    """One arm's sampling configuration, hashed into row identity.

    `temperature=0.0` does not promise bitwise reproducibility. vLLM's batching
    makes greedy decoding sensitive to batch composition, so identical inputs
    can differ across batch sizes. The seed is recorded and treated as part of
    the row's identity precisely because determinism cannot be promised.
    """

    template_id: str = "direct_v1"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    # `None` means "backend default" and is distinct from any numeric value,
    # so it is hashed as null rather than coerced.
    top_k: int | None = None
    min_p: float | None = None
    stop: tuple[str, ...] = ()
    # Whether the backend is given an explicit seed at all. The seed *value*
    # is per-row identity and is not hashed here; whether seeding happens is a
    # property of the configuration and is.
    seeded: bool = True

    @property
    def template(self) -> PromptTemplate:
        return get_template(self.template_id)

    def hash_payload(self) -> dict[str, Any]:
        """The exact dictionary that gets hashed.

        Exposed rather than kept private so a test can assert on it: a silent
        change to what is covered is exactly the failure this guards against,
        and it should be visible in a diff.
        """
        return {
            "v": PARAMS_HASH_VERSION,
            "template_hash": self.template.template_hash,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "stop": list(self.stop),
            "seeded": self.seeded,
        }

    @property
    def params_hash(self) -> str:
        """Twelve hex characters over the canonical payload.

        `sort_keys` plus the tightest separators makes the encoding canonical,
        so the hash does not depend on field declaration order or on how any
        Python version happens to space its JSON.
        """
        blob = json.dumps(
            self.hash_payload(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Full serialization, including the label the hash omits."""
        d = self.hash_payload()
        d["template_id"] = self.template_id
        d["params_hash"] = self.params_hash
        return d

    def evolve(self, **changes: Any) -> "GenParams":
        """Return a copy with fields replaced — and therefore a new hash."""
        return replace(self, **changes)


# Greedy decoding. The reference configuration for a frozen-ladder sweep: one
# sample per (task, arm, seed), no sampling noise beyond what batching imposes.
GREEDY = GenParams(temperature=0.0, top_p=1.0)

# Sampling configuration for the multi-sample arms — best-of-n and the
# self-consistency probe. Temperature has to be non-zero for k samples to
# differ at all, and 0.8 / 0.95 is the conventional operating point for code.
SAMPLED = GenParams(temperature=0.8, top_p=0.95)
