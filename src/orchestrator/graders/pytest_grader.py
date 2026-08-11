"""Execute model-written Python against the task's pytest suite.

SECURITY: the code being run is model output. The `subprocess` backend is a
convenience for local iteration only — it gives process and filesystem-cwd
separation, nothing more. Anything running untrusted or externally-sourced
tasks should use the `docker` backend, which adds a container boundary,
a dropped network, and memory/pid caps.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..types import Grade, Task
from .base import extract_code

# Emitted by the in-sandbox runner so we can recover per-test counts even when
# pytest's exit code alone would only tell us pass/fail.
_REPORT = "__orch_report.json"

_RUNNER = '''
import json, sys
import pytest

class _Collect:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

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
    {"passed": c.passed, "failed": c.failed, "errors": c.errors, "exit": int(code)},
    open("__orch_report.json", "w"),
)
'''


class PytestGrader:
    """Runs the task's tests against the extracted solution module."""

    def __init__(self, timeout_s: float = 60.0, backend: str = "subprocess",
                 image: str = "python:3.12-slim") -> None:
        self.timeout_s = timeout_s
        self.backend = backend
        self.image = image

    def grade(self, task: Task, output: str) -> Grade:
        code = extract_code(output)
        if not code:
            return Grade(passed=0, total=1, error="empty model output")

        workdir = Path(tempfile.mkdtemp(prefix="orch-grade-"))
        try:
            (workdir / "solution.py").write_text(code)
            (workdir / "test_solution.py").write_text(task.tests)
            (workdir / "_runner.py").write_text(_RUNNER)

            t0 = time.perf_counter()
            try:
                proc = self._exec(workdir)
            except subprocess.TimeoutExpired:
                return Grade(passed=0, total=1,
                             error=f"timeout after {self.timeout_s}s",
                             duration_s=self.timeout_s)
            duration = time.perf_counter() - t0

            report_path = workdir / _REPORT
            if not report_path.exists():
                # The runner never got far enough to write a report — almost
                # always a syntax error or an import-time crash in solution.py.
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                return Grade(passed=0, total=1,
                             error=f"harness did not report (exit {proc.returncode}): {tail}",
                             stdout=(proc.stdout or "")[-2000:],
                             duration_s=duration)

            report = json.loads(report_path.read_text())
            passed, failed = report["passed"], report["failed"]
            total = passed + failed
            if total == 0:
                return Grade(passed=0, total=1,
                             error="test suite collected zero tests",
                             stdout=(proc.stdout or "")[-2000:],
                             duration_s=duration)

            return Grade(
                passed=passed,
                total=total,
                error="\n".join(report["errors"])[:4000] or None,
                stdout=(proc.stdout or "")[-2000:],
                duration_s=duration,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _exec(self, workdir: Path) -> subprocess.CompletedProcess:
        if self.backend == "docker":
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "512m",
                "--pids-limit", "128",
                "--cpus", "1",
                "--read-only",
                "--tmpfs", "/tmp:rw,size=64m",
                "-v", f"{workdir}:/work:rw",
                "-w", "/work",
                self.image,
                "sh", "-c", "pip install -q pytest >/dev/null 2>&1 || true; python _runner.py",
            ]
            env = None
        else:
            cmd = [sys.executable, "_runner.py"]
            # Minimal environment: no inherited credentials reach the sandbox.
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(workdir),
                "PYTHONDONTWRITEBYTECODE": "1",
            }

        return subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            env=env,
        )
