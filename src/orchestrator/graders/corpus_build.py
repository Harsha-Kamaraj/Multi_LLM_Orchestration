"""Ingest EvalPlus (HumanEval+ / MBPP+) into R1's task manifest format.

**The `+` is the point** (diya.md): the original HumanEval/MBPP tests are
weak enough that wrong solutions pass. EvalPlus's contribution is a much
larger, auto-generated `plus_input` set that catches what the original few
asserts miss. That asymmetry *is* the visible/hidden split this project
needs — nothing invented, just assigned:

    visible_tests = base_input   (the original, weak assertions)
    tests         = base_input + plus_input   (the full rigorous suite)

EvalPlus ships `base_input`/`plus_input` as raw argument lists plus a
`canonical_solution`, not a ready-to-run pytest module, and its own internal
comparison oracle (float tolerance, special-cased types) is more machinery
than this project needs. Instead: execute the dataset's own
`canonical_solution` — trusted reference code, not model output — once per
input to get the expected value, then emit a plain `assert candidate(*args)
== expected` (tolerance-aware for floats via `atol`) pytest module. Self
contained, inspectable, and it never touches evalplus's comparison internals.

Deterministic selection, not random sampling — same reasoning R1's
`build_corpus(limit=...)` already uses: a pilot has to be repeatable.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from .errors import GraderError

_CLOSE_HELPER = """\
def _close(a, b, atol):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= atol
    return a == b
"""

_MIN_PROMPT_CHARS = 40


def _fetch(dataset: str) -> dict[str, dict[str, Any]]:
    if dataset == "humaneval+":
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()
    if dataset == "mbpp+":
        from evalplus.data import get_mbpp_plus
        return get_mbpp_plus()
    raise GraderError(f"unknown dataset {dataset!r}; expected humaneval+ or mbpp+")


def _reference_fn(prompt: str, entry_point: str, canonical_solution: str) -> Callable:
    """`prompt` already opens the function (signature + docstring);
    `canonical_solution` is the indented body that completes it — this is
    exactly how EvalPlus intends the two fields to be joined."""
    ns: dict[str, Any] = {}
    exec(compile(prompt + canonical_solution, "<canonical>", "exec"), ns)  # noqa: S102
    return ns[entry_point]


def _cases(fn: Callable, inputs: list[list[Any]]) -> list[tuple[list[Any], Any]]:
    """Run the trusted reference on each input, dropping any it can't handle.

    A canonical solution failing on one auto-generated `plus_input` edge case
    is a data-quality issue with that one case, not a reason to drop the
    whole task.
    """
    cases = []
    for args in inputs:
        try:
            cases.append((args, fn(*args)))
        except Exception:  # noqa: BLE001 — a bad generated case, not our bug
            continue
    return cases


def _render_test_module(entry_point: str, cases: list[tuple[list[Any], Any]],
                         atol: float, label: str) -> str:
    lines = [
        f"from solution import {entry_point}",
        "",
        "",
        _CLOSE_HELPER,
        "",
        f"def test_{label}():",
        f"    candidate = {entry_point}",
    ]
    for args, expected in cases:
        lines.append(f"    assert _close(candidate(*{args!r}), {expected!r}, {atol!r})")
    return "\n".join(lines) + "\n"


def to_task_record(problem: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    """Convert one EvalPlus problem into a `data/tasks/*.jsonl` record.

    Returns `None` if the canonical solution doesn't survive conversion well
    enough to grade anything — that's a data-quality filter, logged by the
    caller, not an exception.
    """
    task_id = problem["task_id"]
    entry_point = problem["entry_point"]
    atol = float(problem.get("atol") or 0)
    try:
        fn = _reference_fn(problem["prompt"], entry_point, problem["canonical_solution"])
    except Exception:  # noqa: BLE001 — unconvertible, not our bug
        return None

    base_cases = _cases(fn, problem.get("base_input") or [])
    plus_cases = _cases(fn, problem.get("plus_input") or [])
    if not base_cases:
        return None

    label = task_id.replace("/", "_").lower()
    visible_tests = _render_test_module(entry_point, base_cases, atol, f"{label}_visible")
    hidden_tests = _render_test_module(entry_point, base_cases + plus_cases, atol, f"{label}_hidden")

    return {
        "task_id": task_id,
        "prompt": problem["prompt"],
        "entrypoint": entry_point,
        "tests": hidden_tests,
        "visible_tests": visible_tests,
        "metadata": {
            "dataset": dataset,
            "source": "evalplus",
            "n_visible_cases": len(base_cases),
            "n_hidden_cases": len(base_cases) + len(plus_cases),
        },
    }


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


def contamination_filter(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Drop exact/near-duplicate and degenerate records.

    Every drop is logged with a reason — R4 needs this for the leakage audit
    (diya.md), and "filtered silently" is indistinguishable from "never had
    a duplicate" without it.
    """
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, str]] = []
    seen_prompts: dict[str, str] = {}

    for rec in records:
        norm = _normalize_prompt(rec["prompt"])
        if len(norm) < _MIN_PROMPT_CHARS:
            filtered.append({"task_id": rec["task_id"], "reason": "degenerate_prompt"})
            continue
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if digest in seen_prompts:
            filtered.append({
                "task_id": rec["task_id"],
                "reason": f"duplicate_prompt_of:{seen_prompts[digest]}",
            })
            continue
        seen_prompts[digest] = rec["task_id"]
        kept.append(rec)

    return kept, filtered


def _content_hash(records: list[dict[str, Any]]) -> str:
    """Content hash of the corpus as graded — covers visible *and* hidden
    tests, unlike R1's `CorpusView.fingerprint`, which deliberately excludes
    hidden content because R1 never renders it. This hash exists so R2
    regenerating the corpus is detectable even when only hidden tests
    changed."""
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: r["task_id"]):
        for part in (rec["task_id"], rec["prompt"], rec["entrypoint"],
                     rec["visible_tests"], rec["tests"]):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
    return h.hexdigest()[:16]


def build_pilot(out_path: Path | str, datasets: list[str],
                 n_per_dataset: int) -> dict[str, Any]:
    """Fetch, convert, filter, and deterministically select the pilot corpus.

    Selection is first-N by `task_id` *within each dataset*, post-filter —
    not a random sample, so the pilot is reproducible.
    """
    out_path = Path(out_path)
    all_records: list[dict[str, Any]] = []
    unconvertible: list[dict[str, str]] = []

    for dataset in datasets:
        problems = _fetch(dataset)
        for task_id in sorted(problems):
            record = to_task_record(problems[task_id], dataset)
            if record is None:
                unconvertible.append({"task_id": task_id, "reason": "unconvertible"})
                continue
            all_records.append(record)

    kept, filtered = contamination_filter(all_records)
    filtered = unconvertible + filtered

    selected: list[dict[str, Any]] = []
    by_dataset_count: dict[str, int] = {}
    for rec in sorted(kept, key=lambda r: r["task_id"]):
        dataset = rec["metadata"]["dataset"]
        n_so_far = by_dataset_count.get(dataset, 0)
        if n_so_far >= n_per_dataset:
            continue
        by_dataset_count[dataset] = n_so_far + 1
        selected.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        for rec in selected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": datasets,
        "n_per_dataset": n_per_dataset,
        "kept_count": len(selected),
        "by_dataset": dict(sorted(by_dataset_count.items())),
        "content_hash": _content_hash(selected),
        "filtered": filtered,
    }
    manifest_path = out_path.with_name("MANIFEST.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="",
    )
    return manifest
