"""Run identity — `{date}-{git_sha7}-{config_hash6}`.

A run id is the only handle anyone downstream is allowed to hold. Nothing reads
"latest", so this string is what makes a published number traceable back to the
exact code and configuration that produced it.

A dirty worktree stamps `-dirty` and is **non-publishable**. That suffix is not
a warning to be read and ignored — `is_publishable` exists so the check can be
made mechanically, and R4 refuses to report from a dirty run.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Matches the format the pre-commit hook looks for in `exp:` commit bodies, so
# a run id pasted into a commit message validates without reformatting.
RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[0-9a-f]{7}-[0-9a-f]{6}(-dirty)?$")

# Used when the code is not in a git worktree at all — a tarball, or a notebook
# on a rented box. Deliberately not a real-looking sha: it must be obvious in a
# filename that this run cannot be traced to a commit.
_NO_GIT_SHA = "0000000"


def _git(args: list[str], repo: Path) -> str | None:
    """Run a git command, returning None instead of raising.

    Run identity must never be the reason a twelve-hour sweep fails to start.
    A missing git is degraded provenance, which the id records honestly.
    """
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_sha7(repo: Path | None = None) -> str:
    """Short commit sha, or a visibly fake one outside a worktree."""
    repo = Path(repo or Path.cwd())
    sha = _git(["rev-parse", "--short=7", "HEAD"], repo)
    if not sha or not re.fullmatch(r"[0-9a-f]{7,}", sha):
        return _NO_GIT_SHA
    return sha[:7]


def is_dirty(repo: Path | None = None) -> bool:
    """Whether the worktree differs from HEAD, or git is unavailable.

    **Untracked files count.** An untracked module inside the package changes
    what a sweep does while leaving the recorded sha pointing at code that
    never ran. Ignored paths are excluded, which is what keeps this usable —
    `.gitignore` already covers `runs/`, model weights, and caches, so the
    remaining untracked files are genuinely new source.

    Unavailable git also counts as dirty. The failure mode worth avoiding is a
    run that cannot prove it was clean being treated as if it had been.
    """
    repo = Path(repo or Path.cwd())
    status = _git(["status", "--porcelain", "--untracked-files=normal"], repo)
    if status is None:
        return True
    return bool(status.strip())


def config_hash6(config: dict[str, Any]) -> str:
    """Six hex characters over the canonical encoding of a run's config.

    Canonical means sorted keys and no incidental whitespace, so the same
    configuration hashes identically regardless of how it was assembled.
    Non-serializable values are stringified rather than raising: a config that
    cannot be hashed must not stop a sweep, and `default=str` keeps the value
    contributing to the hash instead of being dropped.
    """
    blob = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:6]


def make_run_id(config: dict[str, Any], repo: Path | None = None,
                now: datetime | None = None) -> str:
    """Build the run id for a sweep about to start.

    `now` is injectable so a test can assert on the whole string rather than on
    a prefix, which is the only way to test the format itself.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    run_id = f"{stamp}-{git_sha7(repo)}-{config_hash6(config)}"
    if is_dirty(repo):
        run_id += "-dirty"
    return run_id


def is_valid(run_id: str) -> bool:
    """Whether a string is a well-formed run id."""
    return bool(RUN_ID_RE.fullmatch(run_id))


def is_publishable(run_id: str) -> bool:
    """Whether a run may back a reported number.

    False for a dirty worktree and false for a run generated outside git.
    Both mean the code that produced the rows cannot be recovered, which is
    exactly the property a published number needs.
    """
    if not is_valid(run_id):
        return False
    return not run_id.endswith("-dirty") and _NO_GIT_SHA not in run_id
