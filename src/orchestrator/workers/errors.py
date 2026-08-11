"""Failure modes R1 must distinguish.

The distinction that matters is *recoverable* versus *poisoning*. A backend
that times out on one request costs one row. A params drift or an unsealed
run silently corrupts every number computed downstream, so those raise
loudly and early rather than degrading.
"""

from __future__ import annotations


class WorkerError(Exception):
    """Base for every error raised inside the workers package."""


class BackendError(WorkerError):
    """A generation backend failed to produce a usable response.

    Recoverable: the sweep records a failed cell and continues. The row is
    still written, with `finish_reason == "error"`, so a re-run can tell
    "not attempted" apart from "attempted and failed".
    """


class BackendUnavailable(WorkerError):
    """A backend was requested but its runtime dependency is not installed.

    Distinct from `BackendError` because it is a setup problem, not a
    generation problem — retrying will never help.
    """


class ParamsDriftError(WorkerError):
    """A cell was found whose `params_hash` disagrees with the current config.

    This is the error that stops a mid-sweep prompt tweak from silently
    poisoning a run. Never downgrade it to a warning.
    """


class StoreError(WorkerError):
    """The rollout store was asked to do something that breaks its contract.

    Mutating a written row, writing to a sealed run, or reading an unsealed
    one. All three are contract violations rather than transient failures.
    """


class UnsealedRunError(StoreError):
    """A reader was pointed at a run with no `_MANIFEST.json`.

    A run directory is invalid until it is sealed; an unsealed run is either
    in flight or was interrupted, and its row count cannot be trusted.
    """


class CharacterizationError(WorkerError):
    """The cost/latency characterization pass could not produce a valid fit.

    Raised when the fit is rank-deficient or the sample is too small to be
    meaningful. A bad fit must fail rather than emit coefficients that look
    plausible and are fiction.
    """
