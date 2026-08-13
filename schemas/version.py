"""The schema version, and the rules that govern changing it.

Every rollout row carries `schema_version`. That single field is what makes a
mixed-version store *detectable* rather than silently averaged — which is the
failure this package exists to prevent. A store containing rows written before
and after a field changed meaning will happily produce a mean. That mean is
not wrong in any way a test would catch.

Changing this number is governed by CONTRIBUTING.md § Schema changes:

* Adding an **optional** field is additive. Old readers ignore it, old rows
  validate against the new schema. One reviewer, no version bump.
* Renaming, removing, retyping, or changing the nullability of a field is
  destructive. It silently breaks readers. All four roles sign off, and the
  version bumps in the same commit.

The distinction is reversibility, not size.
"""

from __future__ import annotations

# Bumped only on a destructive change. Kept in lockstep with
# `orchestrator.workers.generation.SCHEMA_VERSION` — the conformance test in
# `schemas/tests` fails if they drift, because two sources of truth for a
# contract is the same as none.
SCHEMA_VERSION = 1

# Versions this codebase can still read. A row outside this set is refused at
# load rather than coerced: silently reading a v1 row as v2 is precisely the
# mixed-store bug, arriving through the door marked "compatibility".
SUPPORTED_VERSIONS: frozenset[int] = frozenset({1})


class SchemaVersionError(ValueError):
    """A row carries a schema version this codebase cannot read."""


def check_version(version: object, *, where: str = "row") -> int:
    """Validate one row's `schema_version`, or raise.

    Missing is an error rather than a default. A row without a version is a row
    written by code that predates the contract, and guessing which version it
    meant is how a mixed store becomes undetectable.
    """
    if version is None:
        raise SchemaVersionError(
            f"{where} has no schema_version; refusing to guess. "
            f"Rows without a version predate the contract and must be regenerated."
        )
    if isinstance(version, bool) or not isinstance(version, int):
        raise SchemaVersionError(
            f"{where} has non-integer schema_version {version!r}"
        )
    if version not in SUPPORTED_VERSIONS:
        raise SchemaVersionError(
            f"{where} has schema_version {version}, but this codebase supports "
            f"{sorted(SUPPORTED_VERSIONS)}. Re-run the sweep or pin an older "
            f"revision — do not mix versions in one analysis."
        )
    return version


def assert_single_version(rows: object) -> int:
    """Return the one schema version present across `rows`, or raise.

    Called at load time by every consumer. A store spanning two versions is the
    thing no downstream statistic can detect on its own, so it is caught here,
    once, loudly.
    """
    # Mixing is checked *before* support, deliberately. A store containing v1
    # and v2 rows on a codebase that reads only v1 is both problems at once,
    # and "unsupported version 2" would send the reader off to upgrade a
    # reader when the actual defect is that two experiments share a directory.
    # The more specific diagnostic wins.
    seen: set[int] = set()
    n = 0
    for i, row in enumerate(rows):  # type: ignore[call-overload]
        n += 1
        version = row.get("schema_version")
        if version is None or isinstance(version, bool) or not isinstance(version, int):
            check_version(version, where=f"row {i}")  # raises with the right message
        seen.add(version)  # type: ignore[arg-type]

    if not seen:
        raise SchemaVersionError("no rows; cannot determine schema version")
    if len(seen) > 1:
        raise SchemaVersionError(
            f"store mixes schema versions {sorted(seen)} across {n} rows. "
            f"Analyses over a mixed store are invalid. Partition by "
            f"schema_version and re-run, or regenerate the older rows."
        )
    return check_version(seen.pop(), where="store")
