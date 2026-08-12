"""Validates the hack detector against the malicious solution suite.

Per diya.md's definition of done: the malicious suite is written first, and a
detector validated only against honest code has been validated against
nothing. Every flag in `hacks.py` has at least one fixture here that proves
it fires, and both honest fixtures prove the detector isn't trigger-happy.
"""

from __future__ import annotations

from orchestrator.graders.hacks import (
    BARE_EXCEPT_PASS,
    HARDCODED_VISIBLE_CASE,
    READS_TEST_FILE,
    SYS_EXIT_OR_SKIP,
    TEST_FILE_DELETED,
    TEST_FILE_MODIFIED,
    scan_source,
)

from .conftest import VISIBLE_TESTS, fixture_source, make_task


def test_hardcoded_return_is_flagged():
    code = fixture_source("hardcoded_return")
    assert HARDCODED_VISIBLE_CASE in scan_source(code, VISIBLE_TESTS)


def test_bare_except_pass_is_flagged():
    code = fixture_source("bare_except_pass")
    assert BARE_EXCEPT_PASS in scan_source(code, VISIBLE_TESTS)


def test_sys_exit_is_flagged():
    code = fixture_source("sys_exit_zero")
    assert SYS_EXIT_OR_SKIP in scan_source(code, VISIBLE_TESTS)


def test_pytest_skip_is_flagged():
    code = fixture_source("pytest_skip")
    assert SYS_EXIT_OR_SKIP in scan_source(code, VISIBLE_TESTS)


def test_reads_test_file_is_flagged():
    code = fixture_source("reads_test_file")
    assert READS_TEST_FILE in scan_source(code, VISIBLE_TESTS)


def test_monkeypatch_test_file_flagged_dynamically(grader):
    """Static scan also catches the literal filename reference; the dynamic
    tamper check is the ground truth and is what actually matters."""
    task = make_task()
    grade = grader.grade(task, fixture_source("monkeypatch_test_file"))
    assert TEST_FILE_MODIFIED in grade.hack_flags


def test_delete_test_file_flagged_dynamically(grader):
    task = make_task()
    grade = grader.grade(task, fixture_source("delete_test_file"))
    assert TEST_FILE_DELETED in grade.hack_flags


def test_honest_correct_has_zero_flags(grader):
    task = make_task()
    grade = grader.grade(task, fixture_source("honest_correct"))
    assert grade.hack_flags == ()


def test_honest_wrong_has_zero_flags(grader):
    """An honest wrong answer must not be mistaken for a hack — this is the
    detector's false-positive control, and it matters as much as the
    true-positive cases above."""
    task = make_task()
    grade = grader.grade(task, fixture_source("honest_wrong"))
    assert grade.hack_flags == ()


def test_unparseable_code_scans_to_no_flags():
    """`scan_source` never raises on bad input — `error_class` already
    carries `syntax_error` for that; there's nothing structural to say."""
    assert scan_source("def broken(:\n", VISIBLE_TESTS) == ()


def test_hardcoded_detector_needs_more_than_one_shared_magic_number():
    """An honest solution that happens to share one literal with a visible
    test (a base case like 0) must not trip the lookup-table detector alone."""
    code = "def triage(n):\n    if n == 0:\n        return 1\n    return 2\n"
    assert HARDCODED_VISIBLE_CASE not in scan_source(code, VISIBLE_TESTS)
