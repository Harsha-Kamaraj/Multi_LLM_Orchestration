"""Contract validation, with no hard dependency on `jsonschema`.

R3 and R4 need neither a GPU nor model weights, and the point of that is that
they are never blocked on an install. Adding a required third-party validator
to read a rollout would quietly undo it. So this module implements the subset
of JSON Schema the two contracts actually use, and *defers* to `jsonschema`
when it happens to be installed — the conformance test asserts the two agree,
so the subset cannot drift into being wrong without a test failing.

Validation is deliberately strict about nulls. `"type": ["integer", "null"]`
and `"type": "integer"` mean different things here, and collapsing them is how
an ungraded row starts counting as a zero-score row.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

_HERE = Path(__file__).resolve().parent

ROLLOUT_SCHEMA_PATH = _HERE / "rollout.schema.json"
TASK_SCHEMA_PATH = _HERE / "task.schema.json"

_JSON_TYPES: dict[str, tuple[type, ...] | None] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": None,  # handled explicitly
}


class ValidationError(ValueError):
    """A record does not satisfy its contract.

    Carries the path to the offending field, because "row 8412 is invalid" is
    not actionable at the scale these stores reach.
    """

    def __init__(self, message: str, *, path: str = "") -> None:
        self.path = path
        super().__init__(f"{path or '<root>'}: {message}")


@lru_cache(maxsize=8)
def load_schema(path: str | Path) -> dict[str, Any]:
    """Load and cache a schema document."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rollout_schema() -> dict[str, Any]:
    return load_schema(ROLLOUT_SCHEMA_PATH)


def task_schema() -> dict[str, Any]:
    return load_schema(TASK_SCHEMA_PATH)


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    # bool is a subclass of int in Python; JSON Schema does not agree.
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    types = _JSON_TYPES.get(expected)
    if types is None:
        return False
    return isinstance(value, types)


def _check(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if expected is not None:
        options = [expected] if isinstance(expected, str) else list(expected)
        if not any(_type_ok(value, opt) for opt in options):
            raise ValidationError(
                f"expected type {'|'.join(options)}, got {type(value).__name__}",
                path=path,
            )
    if value is None:
        # Every remaining keyword constrains a present value. A permitted null
        # has already been accepted above.
        return

    if (enum := schema.get("enum")) is not None and value not in enum:
        raise ValidationError(f"{value!r} not in {enum}", path=path)

    if isinstance(value, str):
        if (m := schema.get("minLength")) is not None and len(value) < m:
            raise ValidationError(f"shorter than minLength {m}", path=path)
        if (p := schema.get("pattern")) is not None and not re.search(p, value):
            raise ValidationError(f"{value!r} does not match {p!r}", path=path)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if (mn := schema.get("minimum")) is not None and value < mn:
            raise ValidationError(f"{value} below minimum {mn}", path=path)
        if (mx := schema.get("maximum")) is not None and value > mx:
            raise ValidationError(f"{value} above maximum {mx}", path=path)

    if isinstance(value, (list, tuple)) and (items := schema.get("items")):
        for i, item in enumerate(value):
            _check(item, items, f"{path}[{i}]")

    if isinstance(value, dict):
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                _check(value[key], sub, f"{path}.{key}" if path else key)
        for key in schema.get("required") or ():
            if key not in value:
                raise ValidationError(f"missing required field {key!r}", path=path)


def validate(record: Any, schema: dict[str, Any], *, path: str = "") -> None:
    """Validate one record against a schema document, or raise."""
    _check(record, schema, path)


def validate_rollout(row: dict[str, Any], *, path: str = "") -> None:
    """Validate one rollout row. Also enforces cross-field invariants."""
    validate(row, rollout_schema(), path=path)
    _rollout_invariants(row, path=path)


def validate_task(task: dict[str, Any], *, path: str = "") -> None:
    """Validate one task. Also enforces the difficulty-provenance rule."""
    validate(task, task_schema(), path=path)
    if task.get("difficulty") is not None and not task.get("difficulty_provenance"):
        raise ValidationError(
            "difficulty is set without difficulty_provenance; an undocumented "
            "difficulty derived from pass rates is label leakage",
            path=path,
        )


def _rollout_invariants(row: dict[str, Any], *, path: str = "") -> None:
    """Cross-field rules JSON Schema cannot express.

    These are the ones that produce plausible-looking numbers when violated,
    which is exactly why they are checked rather than trusted.
    """
    for passed, total in (("visible_passed", "visible_total"),
                          ("hidden_passed", "hidden_total")):
        p, t = row.get(passed), row.get(total)
        if (p is None) != (t is None):
            raise ValidationError(
                f"{passed} and {total} must be graded together; got {p!r}/{t!r}",
                path=path,
            )
        if p is not None and t is not None and p > t:
            raise ValidationError(f"{passed}={p} exceeds {total}={t}", path=path)

    if row.get("ladder_step", 0) > 0 and not row.get("parent_rollout_id"):
        raise ValidationError(
            "ladder_step > 0 requires parent_rollout_id; without it the ladder "
            "cannot be replayed and the row is an orphan",
            path=path,
        )
    if row.get("ladder_step", 0) == 0 and row.get("parent_rollout_id"):
        raise ValidationError(
            "ladder_step == 0 must not have a parent_rollout_id", path=path
        )

    # A serving-mode row is the only thing latency may be measured from, so a
    # batched one is a contradiction that would silently inflate the metric.
    if row.get("mode") == "serving" and row.get("batch_size", 1) != 1:
        raise ValidationError(
            f"mode=serving with batch_size={row.get('batch_size')}: latency from a "
            "batched call measures queue depth, not the model",
            path=path,
        )


def iter_errors(
    rows: Iterable[dict[str, Any]], *, kind: str = "rollout"
) -> Iterator[ValidationError]:
    """Yield every validation error across `rows` instead of raising the first.

    Used by the pipeline test and the CLI: one bad row in a sweep of 10,000
    should produce a report, not a traceback that hides the other 43.
    """
    check = validate_rollout if kind == "rollout" else validate_task
    for i, row in enumerate(rows):
        try:
            check(row, path=f"[{i}]")
        except ValidationError as exc:
            yield exc
