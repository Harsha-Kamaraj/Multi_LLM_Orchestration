"""`orch` — R2's command line: grading and corpus building.

    orch grade run --run-id 2026-08-14-a3f91c2-7d4e08 --tasks data/tasks/
    orch grade one --tasks data/tasks/pilot_200.jsonl --task-id HumanEval/0 --code-file x.py
    orch corpus build --out data/tasks/pilot_200.jsonl

`grade run` is what makes "grade R1's pilot sweep" a one-command action: it
reads R1's sealed generations, joins them against the task manifest, grades
each row, and seals a graded rollouts store under the same `run_id`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..types import Task
from ..workers.corpus import load_tasks
from ..workers.store import read_generations
from ..workers.store import read_manifest as read_generations_manifest
from .pytest_grader import DEFAULT_DOCKER_IMAGE, PytestGrader
from .rollout_store import RolloutStore, read_manifest

log = logging.getLogger("orch.grade")

DEFAULT_RUNS_ROOT = Path("runs")

# One container per row, each capped at --cpus 1 and --memory 512m, so the
# ceiling is cores rather than RAM. Capped well below the core count: the
# host still has to run dockerd and the pytest processes inside each
# container, and oversubscribing makes every row's timeout budget tighter.
DEFAULT_GRADE_WORKERS = max(1, min(16, (os.cpu_count() or 2) // 2))


def _tasks_by_id(tasks_path: Path) -> dict[str, Task]:
    return {t.task_id: t for t in load_tasks(tasks_path)}


def _make_grader(args: argparse.Namespace, *, publishable: bool) -> PytestGrader:
    backend = args.backend
    if publishable and backend != "docker":
        if not args.unsafe_subprocess_local_only:
            raise SystemExit(
                "refusing to grade with --backend subprocess for a run meant "
                "to produce numbers — the subprocess backend must never "
                "produce reported numbers (diya.md). Pass --backend docker, "
                "or --unsafe-subprocess-local-only to override for local "
                "iteration only."
            )
        log.warning(
            "grading with --backend subprocess: this run is NOT publishable "
            "and must never reach R4"
        )
    return PytestGrader(timeout_s=args.timeout_s, backend=backend, image=args.image)


def _graded_in_order(grader: PytestGrader, pairs: list, workers: int):
    """Yield `(generation, grade)` in manifest order, grading in parallel.

    Safe to parallelise because grading is a pure function of `(task, code)`
    and every call gets its own `mkdtemp` workdir plus its own container — the
    isolation the sandbox already provides per row is exactly what makes rows
    independent of each other.

    Order is preserved deliberately. `ThreadPoolExecutor.map` runs ahead but
    yields in input order, so the graded store is written in manifest order and
    is byte-identical whatever order the containers happen to finish in. A
    graded run that differed between two identical invocations would undermine
    the purity this grader claims.
    """
    if workers <= 1:
        for gen, task in pairs:
            yield gen, grader.grade(task, gen.code)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        grades = pool.map(lambda pair: grader.grade(pair[1], pair[0].code), pairs)
        for (gen, _task), grade in zip(pairs, grades):
            yield gen, grade


def cmd_grade_run(args: argparse.Namespace) -> int:
    generations_root = args.runs_root
    # R1's run must already be sealed — grading an in-progress sweep would
    # describe a partial corpus as if it were the whole one.
    read_generations_manifest(generations_root, args.run_id)  # raises if unsealed

    tasks = _tasks_by_id(args.tasks)
    grader = _make_grader(args, publishable=True)

    pairs = []
    n_missing_task = 0
    for gen in read_generations(generations_root, args.run_id):
        task = tasks.get(gen.task_id)
        if task is None:
            n_missing_task += 1
            log.warning("no task manifest entry for %s; skipping", gen.task_id)
            continue
        pairs.append((gen, task))

    workers = max(1, args.workers)
    log.info("grading %d rows with %d worker(s) on the %s backend",
             len(pairs), workers, args.backend)

    store = RolloutStore(generations_root, args.run_id).open()
    n_graded = 0
    try:
        for gen, grade in _graded_in_order(grader, pairs, workers):
            store.append({**gen.to_row(), **grade.to_row()})
            n_graded += 1
            if n_graded % 100 == 0:
                log.info("  graded %d/%d", n_graded, len(pairs))
    finally:
        manifest = store.seal()

    print(f"graded    {n_graded} rows ({n_missing_task} skipped, no task entry)")
    print(f"run_id    {args.run_id}")
    print(f"solved    {manifest['solved_count']}/{manifest['n_rows']}")
    return 0


def cmd_grade_one(args: argparse.Namespace) -> int:
    tasks = _tasks_by_id(args.tasks)
    task = tasks.get(args.task_id)
    if task is None:
        raise SystemExit(f"no task {args.task_id!r} in {args.tasks}")
    code = Path(args.code_file).read_text(encoding="utf-8") if args.code_file else args.code
    if code is None:
        raise SystemExit("pass --code-file or --code")

    grader = _make_grader(args, publishable=False)
    grade = grader.grade(task, code)
    print(json.dumps(grade.to_row(), indent=2))
    return 0


def cmd_corpus_build(args: argparse.Namespace) -> int:
    from .corpus_build import build_pilot

    manifest = build_pilot(
        out_path=args.out,
        datasets=args.datasets,
        n_per_dataset=args.pilot_n // len(args.datasets),
    )
    print(f"wrote     {args.out}")
    print(f"tasks     {manifest['kept_count']} kept, {len(manifest['filtered'])} filtered")
    print(f"hash      {manifest['content_hash']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orch", description="R2 - verifier and data: grading and corpus.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    grade = sub.add_parser("grade", help="grade generations against tasks")
    grade_sub = grade.add_subparsers(dest="grade_command", required=True)

    run_p = grade_sub.add_parser("run", help="grade a sealed sweep run")
    run_p.add_argument("--run-id", required=True)
    run_p.add_argument("--tasks", type=Path, required=True)
    run_p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    run_p.add_argument("--backend", default="docker", choices=["docker", "subprocess"])
    run_p.add_argument("--image", default=DEFAULT_DOCKER_IMAGE)
    run_p.add_argument("--timeout-s", type=float, default=60.0)
    run_p.add_argument("--unsafe-subprocess-local-only", action="store_true")
    run_p.add_argument("--workers", type=int, default=DEFAULT_GRADE_WORKERS,
                       help="rows graded concurrently. Grading is a pure "
                            "function and each row gets its own container, "
                            "so this changes only wall-clock; the store is "
                            "still written in manifest order")
    run_p.set_defaults(func=cmd_grade_run)

    one_p = grade_sub.add_parser("one", help="grade a single (task, code) pair")
    one_p.add_argument("--tasks", type=Path, required=True)
    one_p.add_argument("--task-id", required=True)
    one_p.add_argument("--code-file", type=Path)
    one_p.add_argument("--code")
    one_p.add_argument("--backend", default="subprocess", choices=["docker", "subprocess"])
    one_p.add_argument("--image", default=DEFAULT_DOCKER_IMAGE)
    one_p.add_argument("--timeout-s", type=float, default=60.0)
    one_p.add_argument("--unsafe-subprocess-local-only", action="store_true")
    one_p.set_defaults(func=cmd_grade_one)

    corpus = sub.add_parser("corpus", help="build the task corpus")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)

    build_p = corpus_sub.add_parser("build", help="ingest EvalPlus into a task manifest")
    build_p.add_argument("--datasets", nargs="+", default=["humaneval+", "mbpp+"])
    build_p.add_argument("--pilot-n", type=int, default=200)
    build_p.add_argument("--out", type=Path, default=Path("data/tasks/pilot_200.jsonl"))
    build_p.set_defaults(func=cmd_corpus_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — CLI boundary, report and exit
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
