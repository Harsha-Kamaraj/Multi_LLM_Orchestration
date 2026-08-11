"""`run_id` is the only handle anyone downstream holds."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orchestrator.workers import runid

CLEAN = "2026-08-14-a3f91c2-7d4e08"


def test_format_matches_the_documented_shape():
    run_id = runid.make_run_id({"a": 1}, now=datetime(2026, 8, 14, tzinfo=timezone.utc))
    assert run_id.startswith("2026-08-14-")
    assert runid.is_valid(run_id)


def test_format_matches_what_the_commit_hook_looks_for():
    """`exp:` commits must carry a run_id in the body, validated by
    `.githooks/pre-commit` against this exact shape."""
    import re

    assert re.match(
        r"^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9a-f]{7}-[0-9a-f]{6}", CLEAN
    )
    assert runid.is_valid(CLEAN)


@pytest.mark.parametrize("bad", [
    "", "2026-08-14", "2026-08-14-a3f91c2", "26-08-14-a3f91c2-7d4e08",
    "2026-08-14-A3F91C2-7d4e08", "2026-08-14-a3f91c2-7d4e08-extra",
])
def test_malformed_ids_are_rejected(bad):
    assert not runid.is_valid(bad)


def test_the_config_hash_is_canonical():
    """Key order must not change the hash, or an identical experiment lands in
    two run directories."""
    assert runid.config_hash6({"a": 1, "b": 2}) == runid.config_hash6({"b": 2, "a": 1})


def test_a_different_config_changes_the_hash():
    assert runid.config_hash6({"a": 1}) != runid.config_hash6({"a": 2})


def test_unserializable_config_values_still_contribute():
    """`default=str` keeps the value in the hash instead of dropping it."""
    from pathlib import Path

    assert runid.config_hash6({"p": Path("a")}) != runid.config_hash6({"p": Path("b")})


def test_a_dirty_worktree_is_stamped_and_unpublishable(monkeypatch):
    monkeypatch.setattr(runid, "is_dirty", lambda *a, **k: True)
    monkeypatch.setattr(runid, "git_sha7", lambda *a, **k: "a3f91c2")
    run_id = runid.make_run_id({}, now=datetime(2026, 8, 14, tzinfo=timezone.utc))
    assert run_id.endswith("-dirty")
    assert runid.is_valid(run_id)
    assert not runid.is_publishable(run_id)


def test_a_clean_run_is_publishable():
    assert runid.is_publishable(CLEAN)


def test_a_run_made_outside_git_is_not_publishable():
    """A visibly fake sha, because the code that produced the rows cannot be
    recovered — exactly the property a published number needs."""
    assert not runid.is_publishable("2026-08-14-0000000-7d4e08")


def test_missing_git_counts_as_dirty(monkeypatch, tmp_path):
    """A run that cannot prove it was clean must not be treated as if it had
    been."""
    monkeypatch.setattr(runid, "_git", lambda *a, **k: None)
    assert runid.is_dirty(tmp_path)
    assert runid.git_sha7(tmp_path) == "0000000"


def test_git_failure_never_raises(monkeypatch, tmp_path):
    """Run identity must never be the reason a twelve-hour sweep fails to
    start."""
    def boom(*a, **k):
        raise OSError("git not found")

    monkeypatch.setattr("subprocess.run", boom)
    assert runid.git_sha7(tmp_path) == "0000000"
    assert runid.is_dirty(tmp_path) is True
