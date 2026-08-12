"""Errors raised by the grading package.

A separate root so callers can catch `GraderError` without also catching
`WorkerError` from R1's side — the two packages fail independently.
"""

from __future__ import annotations


class GraderError(Exception):
    """Something about grading itself is unusable — not a graded outcome.

    A task with no tests, a manifest that won't parse, a sandbox that never
    started. Never raised for "the solution was wrong" — that's a `Grade`
    with `passed < total`, not an exception.
    """
