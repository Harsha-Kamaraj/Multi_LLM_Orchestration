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
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

from .errors import GraderError

# A few EvalPlus `plus_input` cases are deliberately extreme (very large
# integers, to stress numeric solutions). Python 3.11+ refuses int<->str
# conversion past 4300 digits by default as a DoS guard against untrusted
# input. Safe to lift here: this affects only this build-time host process,
# never the grading sandbox — model code runs in its own subprocess/container
# with its own interpreter and the default limit fully intact, which is
# exactly where that guard actually matters.
sys.set_int_max_str_digits(0)

_CLOSE_HELPER = """\
def _close(a, b, atol):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True  # exact match; also inf == inf, which abs(a - b) cannot
                          # see, since inf - inf is nan, not 0
        if a != a or b != b:
            return a != a and b != b  # NaN counts as equal to NaN only
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


# Generous but bounded rendered-literal size. A handful of EvalPlus
# `plus_input` cases are deliberately extreme (see `Mbpp/255`, whose
# canonical `list(combinations_with_replacement(l, n))` produced a 2.5 GB
# result on one input) — dropping the case before it becomes a source
# literal keeps the *grading sandbox's own* interpreter safe to compile,
# its int<->str digit guard fully intact for defending against hostile code.
_MAX_REPR_CHARS = 2000

# Whole-task ceiling, not per-case: one process per task keeps spawn
# overhead bounded (≤200 processes for the pilot) rather than one per test
# case (tens of thousands). The cost is coarse-grained — one runaway case
# drops every case for that task, not just itself — but the next task in
# sorted order backfills the pilot's target count, so this is a data-quality
# trade, not a missing task.
_TASK_TIMEOUT_S = 10.0


def _worker_run_cases(prompt: str, entry_point: str, canonical_solution: str,
                       inputs: list[list[Any]], queue: "mp.Queue") -> None:
    """Runs in its own process — trusted dataset code, but still capable of
    exhausting memory or hanging on an extreme auto-generated input, and this
    is the boundary that contains it."""
    try:
        ns: dict[str, Any] = {}
        exec(compile(prompt + canonical_solution, "<canonical>", "exec"), ns)  # noqa: S102
        fn = ns[entry_point]
    except Exception:  # noqa: BLE001 — unconvertible, not our bug
        queue.put(("init_error", []))
        return

    results: list[tuple[int, Any]] = []
    for i, args in enumerate(inputs):
        try:
            result = fn(*args)
        except Exception:  # noqa: BLE001 — a bad generated case, not our bug
            continue
        if len(repr(args)) > _MAX_REPR_CHARS or len(repr(result)) > _MAX_REPR_CHARS:
            continue
        results.append((i, result))
    queue.put(("ok", results))


def _run_reference_once(prompt: str, entry_point: str, canonical_solution: str,
                         inputs: list[list[Any]]) -> dict[int, Any]:
    """Execute the trusted reference on every input, isolated in its own
    process with a hard wall-clock timeout.

    A timed-out or crashed worker is terminated and its cases are dropped
    entirely — the memory or CPU it consumed goes with it. This is the
    difference between "one pathological case wastes 10 seconds" and what
    actually shipped without it: a 2.5 GB test file the build silently wrote
    to disk.
    """
    if not inputs:
        return {}
    queue: "mp.Queue" = mp.Queue()
    proc = mp.Process(
        target=_worker_run_cases,
        args=(prompt, entry_point, canonical_solution, inputs, queue),
    )
    proc.start()
    proc.join(_TASK_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {}
    try:
        status, results = queue.get_nowait()
    except Exception:  # noqa: BLE001 — nothing was put; treat as no results
        return {}
    if status != "ok":
        return {}
    return dict(results)


def _run_reference(prompt: str, entry_point: str, canonical_solution: str,
                    inputs: list[list[Any]]) -> dict[int, Any]:
    """Run the reference twice, in two independent processes, and keep only
    cases that agree.

    A canonical solution built on `tuple(set(...))` over strings is
    perfectly correct and still order-unstable across process boundaries:
    Python randomizes string hashing per-process by default (a real DoS
    defense, not a bug), so `set` iteration order for strings is not
    guaranteed to match between this build-time process and the grading
    sandbox's own subprocess — even the canonical solution can appear to
    fail a test built from a single build-time capture. Running twice and
    keeping only the agreeing cases empirically finds exactly this class of
    non-determinism instead of guessing which types are affected.
    """
    first = _run_reference_once(prompt, entry_point, canonical_solution, inputs)
    if not first:
        return first
    second = _run_reference_once(prompt, entry_point, canonical_solution, inputs)
    return {i: v for i, v in first.items() if i in second and second[i] == v}


def _literal(value: Any) -> str:
    """Render a value as Python source that re-parses to an equal value.

    Plain `repr()` is not that function for every float: `repr(float('inf'))`
    is the three characters `inf`, which is not a Python literal — it's an
    undefined name. A test built from `assert x == {value!r}` on an
    infinite or NaN expected output compiles fine and then fails with
    `NameError: name 'inf' is not defined`, which is indistinguishable from
    an ordinary failing test in the pass/fail count. It looks exactly like
    the model got the answer wrong; it didn't get a chance to.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('-inf')" if value < 0 else "float('inf')"
        return repr(value)
    if isinstance(value, list):
        return f"[{', '.join(_literal(v) for v in value)}]"
    if isinstance(value, tuple):
        if not value:
            return "()"  # `(,)` — the one-element trailing-comma form — is
                          # not valid syntax when there are zero elements
        inner = ", ".join(_literal(v) for v in value)
        return f"({inner},)"
    if isinstance(value, dict):
        inner = ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items())
        return f"{{{inner}}}"
    return repr(value)


def _render_test_module(entry_point: str, cases: list[tuple[list[Any], Any]],
                         atol: float, label: str) -> str:
    # Imported under an alias, always — a handful of HumanEval/MBPP problems
    # define an entry point whose name itself starts with `test_` (e.g.
    # `test_duplicate`). Importing it under its own name would bind a
    # `test_*` name at module scope, and pytest collects *any* such name it
    # finds there, not just functions it defines itself — so it tries to
    # collect the candidate as a second test, fails on its required
    # argument, and inflates the total by one bogus failure on every task
    # whose entry point happens to be named that way.
    lines = [
        f"from solution import {entry_point} as _candidate_fn",
        "",
        "",
        _CLOSE_HELPER,
        "",
        f"def test_{label}():",
        "    candidate = _candidate_fn",
    ]
    for args, expected in cases:
        lines.append(
            f"    assert _close(candidate(*{_literal(args)}), {_literal(expected)}, {atol!r})"
        )
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
    base_input = list(problem.get("base_input") or [])
    plus_input = list(problem.get("plus_input") or [])
    combined = base_input + plus_input

    results = _run_reference(
        problem["prompt"], entry_point, problem["canonical_solution"], combined,
    )
    if not results:
        return None

    base_cases = [(combined[i], results[i]) for i in range(len(base_input)) if i in results]
    plus_cases = [(combined[i], results[i])
                   for i in range(len(base_input), len(combined)) if i in results]
    if not base_cases:
        return None

    label = task_id.replace("/", "_").lower()
    visible_tests = _render_test_module(entry_point, base_cases, atol, f"{label}_visible")
    hidden_tests = _render_test_module(entry_point, base_cases + plus_cases, atol, f"{label}_hidden")

    return {
        "task_id": task_id,
        "dataset": dataset,
        "prompt": problem["prompt"],
        "entrypoint": entry_point,
        "visible_tests": visible_tests,
        "hidden_tests": hidden_tests,
        # Duplicate of hidden_tests, read by R1's *current* `corpus.py`
        # (`_TESTS_KEYS`), which predates R4's schema freeze and doesn't yet
        # look for `hidden_tests`. Both are the same content; drop `tests`
        # once R1's loader reads the frozen field name instead. Until then
        # this is what keeps the sweep unblocked without editing R1's file.
        "tests": hidden_tests,
        "contamination": {
            "filtered": False,
            "reason": None,
            "cutoff_date": None,
            "source_url": "https://github.com/evalplus/evalplus",
        },
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

    # The frozen contract (schemas/task.schema.json) is the authority on
    # what a task row must look like. Validating here, not just trusting the
    # generator, is what "sign off on schemas/" means in practice — a schema
    # drift shows up as a build failure, not a surprise three roles later.
    from schemas import ValidationError, validate_task
    for rec in selected:
        try:
            validate_task(rec)
        except ValidationError as exc:
            raise GraderError(
                f"generated task {rec['task_id']!r} does not conform to "
                f"schemas/task.schema.json: {exc}"
            ) from exc

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
