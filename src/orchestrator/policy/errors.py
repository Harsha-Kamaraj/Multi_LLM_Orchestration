"""Failure modes of the policy seat, each named after what actually went wrong.

Every error in here exists because the alternative is a number. A loader that
shrugs at a missing column, a mixed-version store, or an ungraded run does not
crash — it produces a policy that trained on two thirds of the data, or on rows
whose meaning changed halfway through, and nothing downstream can tell.

Two of these are the load-bearing ones:

* `LeakageError` — something reached for a label from a feature path.
* `SplitError` — something reached for the test split, which R4 opens exactly
  once and R3 never opens at all.

Both are raised from structural guards rather than from checks a future edit
can forget to call.
"""

from __future__ import annotations


class PolicyError(Exception):
    """Base for everything R3 raises."""


class ContractError(PolicyError):
    """A rollout row is not shaped the way R3 reads it.

    Raised for a missing required column, an uncoercible value, or a row that
    claims a schema version this code was not written against. The message
    always names the column and the row, because a bare "invalid row" on a
    6,000-row store is close to useless.
    """


class SchemaVersionError(ContractError):
    """The store mixes schema versions, or uses one this code cannot read.

    A mixed-version store must be *detectable*, never silently averaged — two
    rows whose `visible_total` means different things are two experiments, and
    a mean over both is a number about nothing.
    """


class LeakageError(PolicyError):
    """A label reached a feature path, or a feature reached forward in time.

    Hidden-test outcomes are labels, never features — including transitively,
    via anything derived from pass rates. This is raised by the guards that
    make that structural instead of aspirational.
    """


class SplitError(PolicyError):
    """Something asked for a split it is not entitled to.

    R3 trains on `train` and calibrates on `val`. The test split is R4's, is
    opened exactly once, and is not loadable through this package at all — the
    absence of a flag is the enforcement.
    """


class StoreReadError(PolicyError):
    """The run cannot be read as pinned.

    Covers a missing run directory, an unsealed run (readers skip those, which
    is what stops an interrupted sweep being read as a complete one), and a
    cost sidecar that was pinned but does not exist.
    """


class UngradedRunError(StoreReadError):
    """The run has generations but no grades, so it carries no labels.

    R1 writes the grading columns as nulls for R2 to fill. A run in that state
    is a perfectly valid *sweep* and a useless *training set*, and the
    difference has to be loud — otherwise the first symptom is an AUC of 0.5
    that looks like a modelling failure.
    """
