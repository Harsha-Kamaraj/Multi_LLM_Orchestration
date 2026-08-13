"""The shared contract. Every role imports it; no role may change it alone.

Two documents — `Task` and `Rollout` — plus the fixture generators that make
them testable. Owned by R4 in `CODEOWNERS` because a contract needs a single
custodian, but destructive changes require sign-off from all four roles (see
CONTRIBUTING.md § Schema changes). The friction is the point: in exchange, every
role develops and tests in complete isolation from day 1.

Nothing here imports from `orchestrator.*`. The dependency runs one way — roles
depend on the contract, the contract depends on nobody — so a change in any
role's package cannot break the contract, and this module can be imported by a
consumer that has none of the serving stack installed.
"""

from __future__ import annotations

from .adversarial import CASES, all_cases
from .synth import ARMS, ArmSpec, SynthConfig, SynthResult, generate, iter_rows
from .validate import (
    ROLLOUT_SCHEMA_PATH,
    TASK_SCHEMA_PATH,
    ValidationError,
    iter_errors,
    rollout_schema,
    task_schema,
    validate,
    validate_rollout,
    validate_task,
)
from .version import (
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    SchemaVersionError,
    assert_single_version,
    check_version,
)

__all__ = [
    "ARMS",
    "CASES",
    "ROLLOUT_SCHEMA_PATH",
    "SCHEMA_VERSION",
    "SUPPORTED_VERSIONS",
    "TASK_SCHEMA_PATH",
    "ArmSpec",
    "SchemaVersionError",
    "SynthConfig",
    "SynthResult",
    "ValidationError",
    "all_cases",
    "assert_single_version",
    "check_version",
    "generate",
    "iter_errors",
    "iter_rows",
    "rollout_schema",
    "task_schema",
    "validate",
    "validate_rollout",
    "validate_task",
]
