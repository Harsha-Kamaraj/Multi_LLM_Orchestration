"""R3 — policy and learning.

Given a rollout store, choose an arm that beats the verifier-gated cascade on
the cost–accuracy frontier.

The input is the rollout store and nothing else that carries an outcome. No
model is called here, no container is run, and the test split is not reachable
through this package.

Three properties are enforced in code rather than in review:

**Labels cannot reach a feature path.** The loader returns rows with the
hidden-test columns physically removed and hands labels back separately, keyed
by `rollout_id`. A feature builder is never given the object that holds them.

**The test split is not loadable.** There is no flag for it. R4 opens it once,
after pre-registration; R3 opening it at all would remove the reason R3 and R4
are two people.

**Nothing resolves "latest".** Every read pins an explicit `run_id`, and a
costing is pinned by coefficient fingerprint the same way.
"""

from __future__ import annotations

__all__ = [
    "PolicyError",
    "ContractError",
    "SchemaVersionError",
    "LeakageError",
    "SplitError",
    "StoreReadError",
    "UngradedRunError",
]

from .errors import (
    ContractError,
    LeakageError,
    PolicyError,
    SchemaVersionError,
    SplitError,
    StoreReadError,
    UngradedRunError,
)
