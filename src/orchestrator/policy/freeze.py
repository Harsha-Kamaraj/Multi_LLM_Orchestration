"""Sealing R3's submission before R4 opens the test split.

`docs/PREREGISTRATION.md` exists to be committed before the frozen split is
opened, and names one command that will be run exactly once. This module is the
R3-shaped half of that: everything the policy contributes to that command,
pinned by content hash, so "nothing changed after the split was opened" is
checkable rather than promised.

## Why a promise is not enough

R3 and R4 are two people specifically so that the person whose policy is being
measured does not control the measurement. That separation survives right up
until the moment R3 can edit a feature builder, refit, and hand over a new
`policy.pkl` under the same name — at which point the pre-registration is a
document about a policy that no longer exists.

Nobody would do that deliberately. It happens by accident: a bug is found in a
feature after the freeze, it is obviously a bug, fixing it obviously improves
things, and the fix is obviously fine because the test split has not been
*looked* at yet. Every step is reasonable and the result is a test measurement
of a policy that was chosen partly by knowing the test set exists.

So the freeze records hashes. Refitting after it produces a different hash, and
`verify` says so. The point is not to prevent the refit — it is to make the
report say which policy the numbers came from.

## What is pinned

The artifacts, and the constants that decide what the artifacts mean:

* `feature_spec.json` — the leakage surface R4 audits independently
* `policy.pkl` and `calibration.json` — the fitted heads and their maps
* `decisions.jsonl` — the actions replayed on the frozen split
* the λ grid, the gate thresholds, and the ECE target

A hash over the artifacts alone would miss the last group entirely: the same
`policy.pkl` read with a different λ grid produces a different frontier, and
nothing in the file would have changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .calibration import ECE_TARGET
from .decide import LAMBDA_GRID
from .errors import PolicyError
from .gate import THRESHOLDS
from .heads import CALIBRATION_FILE, FEATURE_SPEC_FILE, POLICY_PICKLE

FREEZE_FILE = "FREEZE.json"

#: Artifacts hashed when present. A missing one is recorded as missing rather
#: than skipped — "there was no decisions file" and "the decisions file was
#: identical" are different states and a freeze that conflated them would
#: verify happily against a submission that had lost half its contents.
PINNED_FILES: tuple[str, ...] = (
    POLICY_PICKLE,
    CALIBRATION_FILE,
    FEATURE_SPEC_FILE,
    "heads.json",
    "decisions.jsonl",
    "decisions_manifest.json",
)


class FreezeError(PolicyError):
    """A freeze cannot be created or verified as asked."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _constants() -> dict[str, Any]:
    """The knobs that change what an artifact means without changing its bytes."""
    return {
        "lambda_grid_sha256": hashlib.sha256(
            json.dumps([float(v) for v in LAMBDA_GRID]).encode()
        ).hexdigest()[:16],
        "n_lambdas": len(LAMBDA_GRID),
        "lambda_min": float(min(LAMBDA_GRID)),
        "lambda_max": float(max(LAMBDA_GRID)),
        "gate_thresholds": dict(THRESHOLDS),
        "ece_target": ECE_TARGET,
    }


@dataclass(frozen=True)
class Freeze:
    """R3's submission, pinned."""

    run_id: str
    decision_point: str
    created_at: str
    files: Mapping[str, str | None]
    constants: Mapping[str, Any]
    publishable: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decision_point": self.decision_point,
            "created_at": self.created_at,
            "publishable": self.publishable,
            "files": dict(self.files),
            "constants": dict(self.constants),
            "note": self.note,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Freeze":
        return Freeze(
            run_id=str(payload["run_id"]),
            decision_point=str(payload["decision_point"]),
            created_at=str(payload["created_at"]),
            files=dict(payload["files"]),
            constants=dict(payload["constants"]),
            publishable=bool(payload.get("publishable", False)),
            note=str(payload.get("note", "")),
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="",
        )
        return path

    @staticmethod
    def load(path: Path | str) -> "Freeze":
        return Freeze.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class FreezeCheck:
    """What `verify` found, itemised."""

    changed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    appeared: tuple[str, ...] = ()
    constants_changed: tuple[str, ...] = ()

    @property
    def intact(self) -> bool:
        return not (self.changed or self.missing or self.appeared
                    or self.constants_changed)

    def summary(self) -> str:
        if self.intact:
            return "[INTACT] every pinned artifact and constant matches"
        lines = ["[BROKEN] the submission is not what was frozen"]
        for label, items in (
            ("changed", self.changed),
            ("missing since the freeze", self.missing),
            ("appeared since the freeze", self.appeared),
            ("constants changed", self.constants_changed),
        ):
            if items:
                lines.append(f"    {label}: {list(items)}")
        lines.append(
            "    This does not have to mean anything was done wrong. It means "
            "the report must name which policy the numbers came from."
        )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "intact": self.intact,
            "changed": list(self.changed),
            "missing": list(self.missing),
            "appeared": list(self.appeared),
            "constants_changed": list(self.constants_changed),
        }


def _hash_directory(directory: Path) -> dict[str, str | None]:
    return {
        name: (sha256_of(directory / name)
               if (directory / name).is_file() else None)
        for name in PINNED_FILES
    }


def freeze_submission(directory: Path | str, *, run_id: str,
                      decision_point: str, publishable: bool = False,
                      created_at: str = "", note: str = "") -> Freeze:
    """Hash everything R3 is submitting from `directory`.

    Refuses a directory with no policy in it. A freeze over nothing verifies
    perfectly and means nothing, which is the worst possible failure for an
    artifact whose entire job is to be checked later.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FreezeError(f"no submission directory at {directory}")

    files = _hash_directory(directory)
    if files.get(POLICY_PICKLE) is None:
        raise FreezeError(
            f"{directory} holds no {POLICY_PICKLE}, so there is no policy to "
            f"freeze. A freeze over an empty directory verifies perfectly and "
            f"guarantees nothing."
        )

    if not created_at:
        from datetime import datetime, timezone
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return Freeze(
        run_id=run_id,
        decision_point=decision_point,
        created_at=created_at,
        files=files,
        constants=_constants(),
        publishable=publishable,
        note=note,
    )


def verify(frozen: Freeze, directory: Path | str) -> FreezeCheck:
    """Re-hash the submission and report every difference.

    Constants are compared too, and separately. The same `policy.pkl` read
    under a different λ grid produces a different frontier while every file
    hash still matches — a check that only looked at bytes would call that
    intact.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FreezeError(f"no submission directory at {directory}")

    now = _hash_directory(directory)
    changed, missing, appeared = [], [], []
    for name in PINNED_FILES:
        was, is_now = frozen.files.get(name), now.get(name)
        if was is None and is_now is not None:
            appeared.append(name)
        elif was is not None and is_now is None:
            missing.append(name)
        elif was != is_now:
            changed.append(name)

    current = _constants()
    constants_changed = [
        key for key in sorted(set(current) | set(frozen.constants))
        if current.get(key) != frozen.constants.get(key)
    ]

    return FreezeCheck(
        changed=tuple(changed),
        missing=tuple(missing),
        appeared=tuple(appeared),
        constants_changed=tuple(constants_changed),
    )
