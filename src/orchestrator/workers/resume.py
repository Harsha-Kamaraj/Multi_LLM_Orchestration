"""Resume — which cells are already done, keyed so a stale one cannot pass.

Sweeps take hours and will be interrupted: a preempted spot instance, an OOM on
the second model, a laptop closing. Resume has to be exact in both directions.
Recomputing a finished cell wastes GPU hours; accepting a cell computed under
different parameters silently merges two experiments into one store, and
nothing downstream can tell.

The key is **`(task_id, arm, seed, params_hash)`**. Including `params_hash` is
what makes the second failure impossible: a template edit changes the hash, the
old cell no longer matches, and it is regenerated rather than reused.

Errors are retried; refusals and truncations are not. A `finish_reason` of
`error` means the backend never answered — a timeout, a dropped connection —
and is worth another attempt. `refusal` and `length` are real outcomes the
model produced, and retrying them until they change would quietly select for
the lucky sample and bias the arm's measured accuracy upward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import ParamsDriftError
from .store import RolloutStore, read_rows

CellKey = tuple[str, str, int, str]

# Outcomes that count as done. Anything else — currently only `error` — is
# regenerated on resume.
_TERMINAL = frozenset({"stop", "length", "refusal"})


@dataclass
class ResumeIndex:
    """Completed cells for one run, plus what was found while scanning."""

    run_id: str
    completed: set[CellKey] = field(default_factory=set)
    #: Cells present but retryable (`finish_reason == "error"`).
    retryable: set[CellKey] = field(default_factory=set)
    #: params_hash values seen per arm. More than one is a contract violation.
    params_by_arm: dict[str, set[str]] = field(default_factory=dict)
    n_rows: int = 0
    n_torn: int = 0

    def has(self, key: CellKey) -> bool:
        return key in self.completed

    def __contains__(self, key: object) -> bool:
        return key in self.completed

    def __len__(self) -> int:
        return len(self.completed)

    def check_params(self, expected: dict[str, str]) -> None:
        """Assert every arm's stored rows match the configuration about to run.

        This should be impossible: each arm's `params_hash` is folded into the
        run config, so a parameter change produces a different `run_id` and
        therefore a different directory. Reaching this error means a run
        directory was hand-edited, or two configurations were forced to share
        an id — either way the store is no longer one experiment, and it must
        not be appended to.
        """
        for arm, hashes in sorted(self.params_by_arm.items()):
            want = expected.get(arm)
            if want is None:
                continue
            wrong = sorted(h for h in hashes if h != want)
            if wrong:
                raise ParamsDriftError(
                    f"run {self.run_id} already holds rows for arm {arm!r} under "
                    f"params_hash {wrong} but this sweep is configured for "
                    f"{want!r}. A run_id covers exactly one configuration — "
                    f"start a new run rather than mixing two into this one."
                )
            if len(hashes) > 1:
                raise ParamsDriftError(
                    f"run {self.run_id} holds arm {arm!r} under multiple "
                    f"params_hash values {sorted(hashes)}; the run is already "
                    f"two experiments and cannot be extended"
                )


def build_resume_index(root: Path | str, run_id: str, *,
                       retry_errors: bool = True) -> ResumeIndex:
    """Scan an existing run and report which cells are done.

    Reads the run *unsealed* on purpose — a resumable run is by definition one
    that never got its manifest. A run that does not exist yet yields an empty
    index rather than an error, so the first invocation and a resume take the
    same code path and the resuming path is exercised on every sweep.
    """
    index = ResumeIndex(run_id=run_id)
    store = RolloutStore(root, run_id)
    if not store.exists:
        return index

    for row in read_rows(root, run_id, allow_unsealed=True):
        index.n_rows += 1
        try:
            key: CellKey = (
                str(row["task_id"]), str(row["arm"]),
                int(row["seed"]), str(row["params_hash"]),
            )
        except (KeyError, TypeError, ValueError):
            # A row missing identity is unusable for resume. Counted rather
            # than dropped silently: a non-zero count here means something
            # wrote rows this package did not.
            index.n_torn += 1
            continue

        index.params_by_arm.setdefault(key[1], set()).add(key[3])
        reason = str(row.get("finish_reason") or "unknown")
        if reason in _TERMINAL:
            index.completed.add(key)
            index.retryable.discard(key)
        elif retry_errors:
            index.retryable.add(key)
        else:
            index.completed.add(key)
    return index


def pending(cells: Iterable[CellKey], index: ResumeIndex) -> list[CellKey]:
    """Filter a work list down to what still needs generating.

    Order is preserved. The sweep runner builds cells in a deterministic order,
    and keeping it means an interrupted sweep resumes where it stopped instead
    of jumping around the corpus.
    """
    return [c for c in cells if c not in index.completed]


def summarize(index: ResumeIndex, planned: int) -> dict[str, Any]:
    """A short report for the sweep's opening line and the run manifest."""
    done = sum(1 for _ in index.completed)
    return {
        "run_id": index.run_id,
        "rows_on_disk": index.n_rows,
        "cells_complete": done,
        "cells_retryable": len(index.retryable),
        "cells_planned": planned,
        "cells_remaining": max(0, planned - done),
        "torn_rows_skipped": index.n_torn,
    }
