"""Execute model-written Python against a task's visible and hidden tests.

SECURITY: the code being run is model output. Treat every generation as
hostile — eventually one will be, and it won't announce itself.

The `subprocess` backend gives process and filesystem-cwd separation only.
**It must never produce reported numbers** — that's `graders/cli.py`'s job to
enforce, not just this docstring's. Anything grading real or externally
sourced tasks uses the `docker` backend: `--network none --read-only
--memory 512m --pids-limit 128 --cpus 1`, exactly as specified in diya.md.

**The visible/hidden split is enforced by never writing both to disk at
once.** Two separate sandbox runs, two separate temp directories: the visible
run's `test_solution.py` is the weak/original test source
(`task.metadata["visible_tests"]`), the hidden run's is the full rigorous
suite (`task.tests`). A solution that reads `test_solution.py` at runtime
during the visible pass physically cannot see a hidden assertion — it was
never written there. This is what "enforced in code, not by convention" means
in practice.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..types import Task
from .base import Grade, TestResult
from .errors import GraderError
from .hacks import scan_source, tamper_flags

# Emitted by the in-sandbox runner so per-test counts survive even when
# pytest's exit code alone would only say pass/fail.
_REPORT = "__orch_report.json"

_RUNNER = '''
import json, sys
import pytest

class _Collect:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.collect_errors = 0
        self.errors = []

    def pytest_collectreport(self, report):
        if report.failed:
            self.collect_errors += 1
            self.errors.append(f"collect: {report.longreprtext[:800]}")

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            if report.failed:
                self.failed += 1
                self.errors.append(f"{report.nodeid}: {report.longreprtext[:800]}")
            return
        if report.passed:
            self.passed += 1
        elif report.failed:
            self.failed += 1
            self.errors.append(f"{report.nodeid}: {report.longreprtext[:800]}")

c = _Collect()
code = pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", "test_solution.py"], plugins=[c])
json.dump(
    {
        "passed": c.passed,
        "failed": c.failed,
        "collect_errors": c.collect_errors,
        "errors": c.errors,
        "exit": int(code),
    },
    open("__orch_report.json", "w"),
)
'''

_DOCKER_FLAGS = [
    "--network", "none",
    "--memory", "512m",
    "--pids-limit", "128",
    "--cpus", "1",
    "--read-only",
    "--tmpfs", "/tmp:rw,size=64m",
]

# Built by src/orchestrator/graders/docker/Dockerfile — pytest baked in at
# build time, because `--network none` means the container cannot fetch it
# at run time. Never fall back to installing pytest inside `_exec`; a sandbox
# that tries to pip-install under a dropped network fails silently and every
# task grades as a harness error, which reads as "the sandbox is broken"
# instead of what it is: the wrong image.
DEFAULT_DOCKER_IMAGE = "orch-grader:py312"


class _RunOutcome:
    """One sandbox execution: a test tier's result plus harness health."""

    __slots__ = ("result", "error_class", "duration_s")

    def __init__(self, result: TestResult, error_class: str, duration_s: float) -> None:
        self.result = result
        self.error_class = error_class
        self.duration_s = duration_s


# Harness health outcomes ranked worst-first; the grade's overall
# `error_class` is the worst seen across both runs, since a solution that
# fails to import can't sensibly be "none" just because one tier happened to
# have zero tests to collect.
_SEVERITY = {"none": 0, "runtime_error": 1, "timeout": 2, "harness_error": 3}


class PytestGrader:
    """Runs a task's visible and hidden tests, separately, against `code`."""

    def __init__(self, timeout_s: float = 60.0, backend: str = "subprocess",
                 image: str = DEFAULT_DOCKER_IMAGE) -> None:
        self.timeout_s = timeout_s
        self.backend = backend
        self.image = image

    def grade(self, task: Task, code: str) -> Grade:
        visible_src = str(task.metadata.get("visible_tests", "") or "")
        hidden_src = task.tests or ""

        if not code.strip():
            return Grade(
                visible=TestResult(0, _count_asserts(visible_src)),
                hidden=TestResult(0, _count_asserts(hidden_src)),
                error_class="empty_code",
            )
        try:
            ast.parse(code)
        except (SyntaxError, ValueError):
            return Grade(
                visible=TestResult(0, _count_asserts(visible_src)),
                hidden=TestResult(0, _count_asserts(hidden_src)),
                error_class="syntax_error",
                hack_flags=scan_source(code, visible_src),
            )

        t0 = time.perf_counter()
        hack_flags = set(scan_source(code, visible_src))

        visible_outcome = self._run_tier(code, visible_src, hack_flags) \
            if visible_src.strip() else _RunOutcome(TestResult(0, 0), "none", 0.0)
        hidden_outcome = self._run_tier(code, hidden_src, hack_flags) \
            if hidden_src.strip() else _RunOutcome(TestResult(0, 0), "none", 0.0)

        duration = time.perf_counter() - t0
        error_class = max(
            (visible_outcome.error_class, hidden_outcome.error_class),
            key=lambda e: _SEVERITY[e],
        )

        return Grade(
            visible=visible_outcome.result,
            hidden=hidden_outcome.result,
            error_class=error_class,
            hack_flags=tuple(sorted(hack_flags)),
            duration_s=duration,
        )

    # -- one tier, one fresh sandbox ------------------------------------------

    def _run_tier(self, code: str, tests_src: str, hack_flags: set[str]) -> _RunOutcome:
        workdir = Path(tempfile.mkdtemp(prefix="orch-grade-"))
        try:
            (workdir / "solution.py").write_text(code, encoding="utf-8")
            test_path = workdir / "test_solution.py"
            test_path.write_text(tests_src, encoding="utf-8")
            original_bytes = test_path.read_bytes()
            (workdir / "_runner.py").write_text(_RUNNER, encoding="utf-8")

            t0 = time.perf_counter()
            try:
                proc = self._exec(workdir)
            except subprocess.TimeoutExpired:
                hack_flags.update(tamper_flags(original_bytes, test_path))
                return _RunOutcome(
                    TestResult(0, _count_asserts(tests_src)),
                    "timeout",
                    self.timeout_s,
                )
            duration = time.perf_counter() - t0
            if self.backend == "docker":
                _check_docker_infra(proc)
            tamper = tamper_flags(original_bytes, test_path)
            hack_flags.update(tamper)
            if tamper:
                # The oracle was corrupted mid-run — whatever pytest reports
                # about a test file that no longer matches what was written
                # is not trustworthy signal, and *how* an OS reacts to a
                # file vanishing out from under a running interpreter is
                # exactly the kind of platform detail grading must not
                # depend on. Zero credit, deterministically, without reading
                # the report at all.
                return _RunOutcome(TestResult(0, _count_asserts(tests_src)), "none", duration)

            report_path = workdir / _REPORT
            if not report_path.exists():
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                return _RunOutcome(
                    TestResult(0, _count_asserts(tests_src), errors=tail),
                    "harness_error",
                    duration,
                )

            report = json.loads(report_path.read_text())
            passed, failed = report["passed"], report["failed"]
            total = passed + failed
            errors = "\n".join(report["errors"])[:4000]

            if total == 0:
                # Nothing collected. A model that emptied the test module (or
                # broke collection) looks identical to "no tests exist" from
                # pytest's exit code alone — `collect_errors` disambiguates.
                if report.get("collect_errors"):
                    return _RunOutcome(
                        TestResult(0, _count_asserts(tests_src), errors=errors),
                        "runtime_error",
                        duration,
                    )
                return _RunOutcome(TestResult(0, 0), "none", duration)

            return _RunOutcome(TestResult(passed, total, errors=errors), "none", duration)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _exec(self, workdir: Path) -> subprocess.CompletedProcess:
        if self.backend == "docker":
            cmd = [
                "docker", "run", "--rm",
                *_DOCKER_FLAGS,
                "-v", f"{workdir}:/work:rw",
                "-w", "/work",
                self.image,
                "python", "_runner.py",
            ]
            env = None
        elif self.backend == "subprocess":
            cmd = [sys.executable, "_runner.py"]
            # Minimal environment: no inherited credentials reach the sandbox.
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(workdir),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            if os.name == "nt":
                # Not a credential — required for Windows socket/DLL init
                # (transitive imports like asyncio.windows_events fail
                # without it). The docker backend is unaffected; this only
                # touches local-iteration-only subprocess runs on Windows.
                env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
        else:
            raise GraderError(f"unknown grader backend {self.backend!r}")

        return subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            env=env,
        )


_DOCKER_INFRA_MARKERS = (
    "unable to find image",
    "pull access denied",
    "repository does not exist",
    "cannot connect to the docker daemon",
    "error during connect",  # e.g. Docker Desktop not running, on Windows
    "is not recognized",  # docker CLI itself missing, on Windows
)


def _check_docker_infra(proc: subprocess.CompletedProcess) -> None:
    """Distinguish "the sandbox is misconfigured" from "the code failed".

    A missing image or a dead daemon is not a property of the code being
    graded — it must never silently become a `harness_error` on every row of
    a sweep, which reads as "the model's code is broken" instead of what it
    actually is. Raised once, loudly, instead of graded.
    """
    stderr = (proc.stderr or "").lower()
    if any(marker in stderr for marker in _DOCKER_INFRA_MARKERS):
        raise GraderError(
            f"docker sandbox is not usable ({proc.stderr.strip()[:300]}); "
            f"build the grading image first: docker build -t {DEFAULT_DOCKER_IMAGE} "
            f"-f src/orchestrator/graders/docker/Dockerfile ."
        )


def _count_asserts(tests_src: str) -> int:
    """Best-effort test count for a tier that never got to run.

    Used only when grading short-circuits before the sandbox (empty code,
    a syntax error, a timeout). Counts `def test_*` functions if the module
    parses as pytest-style, falling back to `assert` statements for a bare
    assertion script — either way it's an estimate for `total`, not a claim
    about what pytest would have collected.
    """
    if not tests_src.strip():
        return 0
    try:
        tree = ast.parse(tests_src)
    except (SyntaxError, ValueError):
        return 0
    test_funcs = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    )
    if test_funcs:
        return test_funcs
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))
