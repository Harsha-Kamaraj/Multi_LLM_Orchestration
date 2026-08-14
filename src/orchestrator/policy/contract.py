"""The rollout row, as R3 reads it.

`schemas/` is now the ratified contract, so this module no longer restates it.
Row validity and schema versioning are delegated there — two sources of truth
for a contract is the same as none. What stays here is the part `schemas/` does
not model and R3 owns: **which columns are readable at which decision point.**

Three things this file decides, and they are not stylistic:

**Which columns are labels.** `hidden_passed` and `hidden_total` are outcomes
of tests the model never saw. They are the target and they are never an input,
including transitively — a "difficulty" column derived from pass rates is the
same leak wearing a different name. The loader removes them from the row
entirely rather than trusting anyone to remember.

**Which columns are observable at which decision point.** D0 decides *which arm
to try* and can see only what exists before generating. D1 decides *whether to
escalate* and may additionally see the candidate, its extraction, and the
visible-test outcome. A feature that reads a D1 column while claiming D0 is
the leak that actually happens, so the allowlists are data and Phase 3 asserts
against them.

**Which columns are never a feature at any point.** `wall_ms` is the one that
matters: under `mode == "sweep"` it is a function of queue depth rather than of
the model, so a feature built on it is measuring batch composition. It stays on
the row for auditing and is excluded from both allowlists.

## The prompt is not on the row

Worth stating plainly, because it contradicts the shorthand that R3's input is
"the rollout store and nothing else": **the rollout row carries no prompt.** It
carries `text` and `code` — what the model produced — and the `task_id` that
produced them. Every prompt-only D0 feature (length, visible-test count,
keyword structure) is therefore uncomputable from the store alone. This is a
property of the ratified schema, not of R1's draft, and it is open with R4.

The second input is R2's task manifest, joined by `task_id`. It is a data
artifact, not another role's code, and the join is explicit and prefixed
`task_*` so nothing can confuse a task's prompt with a model's output. Only
`prompt`, `entrypoint`, and `visible_tests` cross the join — `Task.tests`
holds the full suite and is left behind, exactly as R1's `PromptContext`
carries a visible-tests field and no hidden one.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from schemas.version import SUPPORTED_VERSIONS, check_version
from schemas.version import SchemaVersionError as ContractSchemaVersionError

from .errors import ContractError, SchemaVersionError

# Re-exported under R3's older name so callers keep working, but the values
# come from the contract package. R3 does not get an opinion about which
# schema versions exist.
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = SUPPORTED_VERSIONS

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

#: Identity of the cell. `rollout_id` is derived by R1 and is the join key for
#: everything that lands beside a row — cost sidecars, labels, decisions.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "rollout_id", "run_id", "task_id", "arm", "seed", "params_hash",
    "split", "dataset", "code_version", "schema_version",
)

#: Known before a single token is generated. The arm and its resolved model are
#: in here on purpose: `P_pass[a](x)` is fitted per arm, so arm identity indexes
#: the head rather than entering it as a feature.
D0_OBSERVABLE: frozenset[str] = frozenset({
    "task_id", "split", "dataset", "arm", "seed", "params_hash",
    "model_id", "ladder_step", "parent_rollout_id",
    # Joined from R2's manifest. Prefixed so a prompt can never be mistaken for
    # model output, and carrying no hidden tests.
    "task_prompt", "task_entrypoint", "task_visible_tests",
    # Fixture-only: the planted prompt-only proxy, present on a synthetic
    # corpus and absent on a real one. D0-observable because that is exactly
    # what it stands in for — a prompt feature. The latent difficulty it is a
    # noisy view of is quarantined in `RolloutData.latent` and appears in no
    # allowlist at all.
    "task_x_d0",
})

#: Additionally observable once the candidate exists and the visible tests have
#: run. This is the information the D0 → D1 comparison isolates.
D1_ONLY_OBSERVABLE: frozenset[str] = frozenset({
    "text", "code", "code_parses", "extract_strategy",
    "prefill_tokens", "decode_tokens", "finish_reason", "error",
    "visible_passed", "visible_total", "error_class", "hack_flags",
    "grade_duration_s",
    # Derived from the token counts above via the pinned coefficients, so they
    # are knowable exactly when the token counts are. At D0 these are what
    # `E_cost` and `E_latency` predict, never what they read.
    "gpu_seconds", "imputed_latency_s", "usd",
})

D1_OBSERVABLE: frozenset[str] = D0_OBSERVABLE | D1_ONLY_OBSERVABLE

#: Outcomes of tests the model never saw. Target only, at every decision point.
LABEL_COLUMNS: frozenset[str] = frozenset({"hidden_passed", "hidden_total"})

#: Recorded so a run can be audited, and excluded from both allowlists so none
#: of it can drift into a model.
PROVENANCE_COLUMNS: frozenset[str] = frozenset({
    "run_id", "code_version", "schema_version", "created_at",
    "mode", "batch_size", "backend", "tokens_exact", "extra",
})

#: Not a feature at any decision point, for a reason specific to each.
#:
#: `wall_ms` — under `mode == "sweep"` this is queue depth, not model speed. A
#: feature built on it measures how full the batch was.
#: `hidden_*` — labels.
NEVER_A_FEATURE: frozenset[str] = LABEL_COLUMNS | {"wall_ms"}


def observable_at(decision_point: str) -> frozenset[str]:
    """Columns a feature may read at the named decision point."""
    if decision_point == "D0":
        return D0_OBSERVABLE
    if decision_point == "D1":
        return D1_OBSERVABLE
    raise ContractError(
        f"unknown decision point {decision_point!r}; this project has exactly "
        f"two, 'D0' (before generating) and 'D1' (after generating)"
    )


# ---------------------------------------------------------------------------
# Row shape
# ---------------------------------------------------------------------------

#: Absent or unparseable in any of these and the row is not usable at all.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "rollout_id", "task_id", "arm", "seed", "params_hash", "schema_version",
)

_INT_COLUMNS: frozenset[str] = frozenset({
    "seed", "prefill_tokens", "decode_tokens", "batch_size", "ladder_step",
    "schema_version", "visible_passed", "visible_total",
    "hidden_passed", "hidden_total",
})

_FLOAT_COLUMNS: frozenset[str] = frozenset({
    "wall_ms", "gpu_seconds", "imputed_latency_s", "grade_duration_s", "usd",
})

_BOOL_COLUMNS: frozenset[str] = frozenset({"tokens_exact", "code_parses"})

_STR_COLUMNS: frozenset[str] = frozenset({
    "rollout_id", "run_id", "task_id", "arm", "params_hash", "model_id",
    "text", "code", "finish_reason", "mode", "backend", "extract_strategy",
    "split", "dataset", "code_version", "created_at",
})

#: Legitimately null. Everything else missing is a contract violation, not a
#: gap to fill with a zero — imputing a default here is how an ungraded run
#: quietly becomes a training set of all-zero labels.
NULLABLE_COLUMNS: frozenset[str] = frozenset({
    "parent_rollout_id", "error", "gpu_seconds", "imputed_latency_s", "usd",
    "visible_passed", "visible_total", "hidden_passed", "hidden_total",
    "error_class", "hack_flags", "grade_duration_s",
})

#: Written by R2. Their presence is what separates a sweep from a training set.
GRADE_COLUMNS: tuple[str, ...] = (
    "visible_passed", "visible_total", "hidden_passed", "hidden_total",
)


def _coerce_int(value: Any, column: str, where: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        # Via float so a Parquet round-trip that widened an int to 3.0 still
        # reads. A genuinely fractional count is a contract violation and is
        # rejected below rather than silently truncated.
        as_float = float(value)
    except (TypeError, ValueError):
        raise ContractError(
            f"{where}: column {column!r} is {value!r}, which is not an integer"
        ) from None
    if as_float != int(as_float):
        raise ContractError(
            f"{where}: column {column!r} is {value!r}; counts are integers and "
            f"rounding one silently would change what it counts"
        )
    return int(as_float)


def _coerce_float(value: Any, column: str, where: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ContractError(
            f"{where}: column {column!r} is {value!r}, which is not a number"
        ) from None


def _coerce_bool(value: Any, column: str, where: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ContractError(
        f"{where}: column {column!r} is {value!r}, which is not a boolean"
    )


def _coerce_hack_flags(value: Any) -> tuple[str, ...] | None:
    """Normalize R2's hack flags to a tuple, whatever shape they arrive in.

    JSONL gives a list, Parquet may give a list or a JSON string, and an
    ungraded row gives null. All three are legitimate; a mix of tuple and list
    downstream is not.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return (text,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _coerce_extra(value: Any) -> dict[str, Any]:
    """Normalize the backend-specific bag.

    R1 serializes `extra` to a JSON string when writing Parquet because Arrow
    would otherwise infer a struct from the first row and fail on the rest. So
    the same run read through JSONL and through Parquet hands back a dict in
    one path and a string in the other. Normalizing here is what stops the two
    read paths quietly diverging.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_unparsed": value}
        return dict(parsed) if isinstance(parsed, Mapping) else {"_value": parsed}
    return {"_value": value}


def normalize_row(row: Mapping[str, Any], *, where: str = "row") -> dict[str, Any]:
    """Coerce one stored row into the types R3 reads, or say why it cannot.

    `where` is threaded through every message because a bare "invalid row" on a
    six-thousand-row store is close to useless, and the file-and-line form is
    what makes it a two-second fix.
    """
    out: dict[str, Any] = dict(row)

    for column in REQUIRED_COLUMNS:
        if out.get(column) is None or out.get(column) == "":
            raise ContractError(
                f"{where}: required column {column!r} is missing or empty; "
                f"present columns are {sorted(out)}"
            )

    # Delegated to the contract package rather than restated. R3 does not get
    # to hold a second opinion about which versions are readable — that is the
    # mixed-store bug arriving through the door marked "compatibility".
    version = _coerce_int(out["schema_version"], "schema_version", where)
    try:
        out["schema_version"] = check_version(version, where=where)
    except ContractSchemaVersionError as exc:
        raise SchemaVersionError(str(exc)) from None

    for column in _INT_COLUMNS:
        if column in out:
            out[column] = _coerce_int(out[column], column, where)
    for column in _FLOAT_COLUMNS:
        if column in out:
            out[column] = _coerce_float(out[column], column, where)
    for column in _BOOL_COLUMNS:
        if column in out:
            out[column] = _coerce_bool(out[column], column, where)
    for column in _STR_COLUMNS:
        if column in out and out[column] is not None:
            out[column] = str(out[column])

    out["hack_flags"] = _coerce_hack_flags(out.get("hack_flags"))
    out["extra"] = _coerce_extra(out.get("extra"))
    if out.get("parent_rollout_id") in ("", None):
        out["parent_rollout_id"] = None

    return out


def is_graded(row: Mapping[str, Any]) -> bool:
    """Whether R2 has filled this row's grading columns.

    `visible_total` of zero counts as ungraded rather than as a task with no
    tests: a visible-test outcome is the whole basis of the D1 decision, and a
    row that cannot produce one is not a D1 training example.
    """
    hidden_total = row.get("hidden_total")
    visible_total = row.get("visible_total")
    if hidden_total is None or visible_total is None:
        return False
    return int(hidden_total) > 0


def solved(row: Mapping[str, Any]) -> bool:
    """Strict all-or-nothing correctness on the hidden suite — the label.

    All-or-nothing rather than a pass fraction because that is what the project
    reports as accuracy, and a head fitted against partial credit would be
    calibrated to a quantity nothing else in the pipeline uses.
    """
    total = row.get("hidden_total")
    passed = row.get("hidden_passed")
    if not total or passed is None:
        return False
    return int(passed) == int(total)


def assert_no_labels(columns: Iterable[str], *, context: str) -> None:
    """Refuse a column set that contains a label. Raised, never warned.

    Called by the loader after stripping, and by the feature layer before
    building. Two calls rather than one because a guard that runs in only one
    place stops being a guard the moment someone adds a second path.
    """
    from .errors import LeakageError

    offending = sorted(set(columns) & LABEL_COLUMNS)
    if offending:
        raise LeakageError(
            f"{context}: {offending} are hidden-test outcomes and are labels, "
            f"never features — not directly, and not via anything derived from "
            f"them. If a feature needs a difficulty signal, it must come from "
            f"the visible tests or from the prompt."
        )
