"""Every one of R4's adversarial stores, run through R3's loader.

`schemas/adversarial.py` states the bar plainly: *consumers are expected to
reject these, not survive them. "Robust" here means refusing to produce a
number, not producing one anyway.* This file is R3 meeting that bar, case by
case, with the expected outcome written down next to each one.

The table below is the contract. A case that starts passing for a different
reason than the one recorded is not a pass — so each expectation names both the
exception and, where it applies, the specific check that must fire. A loader
that rejected everything with one generic error would satisfy `pytest.raises`
and tell you nothing about which corruption it actually caught.

One case is deliberately *not* an error. A `-dirty` run is readable — you may
develop against it — and merely non-publishable. Turning it into a refusal
would stop work that is legitimate, and the guard that matters is on the
output, not the input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator.policy import store
from orchestrator.policy.errors import (
    ContractError, PolicyError, SchemaVersionError, StoreIntegrityError,
    StoreReadError, UngradedRunError,
)
from schemas.adversarial import CASES, all_cases

#: case name -> (exception, substring that must appear in the message)
#:
#: The substring is what pins the *reason*. Without it, a loader that raised
#: `StoreIntegrityError("something is wrong")` for all twelve would pass.
#:
#: Two layers do the work, and which one fires is recorded here rather than
#: left to chance:
#:
#: * **`rollout.schema.json`**, via R4's validator, owns everything visible in
#:   one row — including the cross-*field* rules it encodes, like `passed`
#:   without `total` and `ladder_step > 0` without a parent.
#: * **`policy/integrity.py`** owns what no single row can show: the same cell
#:   twice, a task in two splits, one arm where there should be two. A
#:   per-row validator cannot see any of these no matter how strict it is.
#:
#: The division is worth keeping straight, because a check migrating from one
#: layer to the other silently changes which exception a caller must catch.
EXPECTED: dict[str, tuple[type[Exception], str]] = {
    # --- caught per row, by the contract ------------------------------------
    "mixed_schema_versions": (SchemaVersionError, "schema_version 2"),
    "half_graded_pair": (ContractError, "graded together"),
    "negative_and_impossible_counts": (ContractError, "exceeds hidden_total"),
    "orphan_ladder_step": (ContractError, "requires parent_rollout_id"),
    "serving_row_batched": (ContractError, "batched"),
    # --- caught across rows, by R3 ------------------------------------------
    "duplicate_rollout_ids": (StoreIntegrityError, "duplicate_rollout_id"),
    "split_leakage": (StoreIntegrityError, "task_spans_splits"),
    "unicode_and_nulls": (StoreIntegrityError, "control_characters"),
    "single_arm_only": (StoreIntegrityError, "single_arm"),
    # --- caught by the loader's own preconditions ---------------------------
    "ungraded_rows_as_zero": (UngradedRunError, "no hidden-test outcome"),
    "unsealed_run": (StoreReadError, "_MANIFEST"),
}

#: The cases R3 must catch on its own, with the contract validator switched
#: off. These are the ones that justify `policy/integrity.py` existing at all.
INTEGRITY_OWNED = {
    "duplicate_rollout_ids": "duplicate_rollout_id",
    "split_leakage": "task_spans_splits",
    "unicode_and_nulls": "control_characters",
    "single_arm_only": "single_arm",
}

#: Readable, and non-publishable. Not a refusal.
TOLERATED = {"dirty_run_id"}


def write_case(root: Path, rows: list[dict[str, Any]], *,
               sealed: bool = True) -> str:
    """Lay a case out on disk exactly as a real run would be."""
    run_id = str(rows[0]["run_id"])
    rows_dir = root / run_id / "generations"
    rows_dir.mkdir(parents=True, exist_ok=True)
    part = rows_dir / "part-000001-0000000000-aaaaaa.jsonl"
    with part.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    if sealed:
        (root / run_id / "_MANIFEST.json").write_text(
            json.dumps({"run_id": run_id, "schema_version": 1,
                        "n_rows": len(rows), "publishable": True}),
            encoding="utf-8",
        )
    return run_id


def test_every_adversarial_case_has_a_recorded_expectation():
    """A new case in `schemas/` must not silently go unhandled by R3.

    This is the test that fails when R4 adds a thirteenth corruption. Without
    it, the suite below would keep passing while quietly covering less.
    """
    assert set(CASES) == set(EXPECTED) | TOLERATED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_loader_refuses_the_case(tmp_path: Path, name: str):
    rows, why = all_cases()[name]
    exception, reason = EXPECTED[name]

    # `unsealed_run` is defined by the *absence* of a manifest, not by its rows.
    run_id = write_case(tmp_path, rows, sealed=(name != "unsealed_run"))

    with pytest.raises(exception) as excinfo:
        store.load_rollouts(tmp_path, run_id)

    message = str(excinfo.value)
    assert reason in message, (
        f"{name} was refused, but for the wrong reason.\n"
        f"  fixture guards: {why}\n"
        f"  expected to mention: {reason!r}\n"
        f"  actually said: {message}"
    )


@pytest.mark.parametrize("name", sorted(INTEGRITY_OWNED))
def test_r3_catches_these_without_help_from_the_validator(
    tmp_path: Path, name: str,
):
    """The cases that justify `policy/integrity.py` existing.

    Run with schema validation off, so the only thing that can refuse them is
    R3's own cross-row checking. A per-row validator cannot see a duplicated
    cell or a task spanning two splits however strict it is.
    """
    rows, why = all_cases()[name]
    run_id = write_case(tmp_path, rows)

    with pytest.raises(StoreIntegrityError) as excinfo:
        store.load_rollouts(tmp_path, run_id, validate_schema=False)

    assert INTEGRITY_OWNED[name] in str(excinfo.value), why


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_refusal_is_a_policy_error(tmp_path: Path, name: str):
    """One hierarchy, so a caller can wrap the whole load in one except."""
    rows, _ = all_cases()[name]
    run_id = write_case(tmp_path, rows, sealed=(name != "unsealed_run"))
    with pytest.raises(PolicyError):
        store.load_rollouts(tmp_path, run_id)


def test_a_dirty_run_is_readable_but_never_publishable(tmp_path: Path):
    """Developing against a dirty run is fine. Reporting from one is not."""
    rows, _ = all_cases()["dirty_run_id"]
    run_id = write_case(tmp_path, rows)
    assert run_id.endswith("-dirty")

    data = store.load_rollouts(tmp_path, run_id)
    assert len(data) > 0
    assert data.publishable is False


def test_a_manifest_cannot_override_a_dirty_run_id(tmp_path: Path):
    """`write_case` writes `publishable: true`; the run_id has the last word."""
    rows, _ = all_cases()["dirty_run_id"]
    run_id = write_case(tmp_path, rows)
    data = store.load_rollouts(tmp_path, run_id)
    assert data.manifest["publishable"] is True
    assert data.publishable is False


def test_the_healthy_fixture_it_is_derived_from_loads_cleanly(tmp_path: Path):
    """The control.

    Every case above is a mutation of one healthy store. If that store did not
    itself load, the twelve refusals would prove nothing — a loader that
    refused all input would pass every test in this file.
    """
    from schemas.synth import SynthConfig, generate

    result = generate(SynthConfig(n_tasks=40, seeds=3), seed=7)
    run_id = write_case(tmp_path, result.rows)

    data = store.load_rollouts(tmp_path, run_id)
    assert len(data) > 0
    assert sorted(data.arms) == ["large", "small"]
    assert data.publishable is True
