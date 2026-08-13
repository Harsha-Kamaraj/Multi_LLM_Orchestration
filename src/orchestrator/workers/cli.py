"""`orch-workers` — the one command.

A sweep that needs a checklist gets run inconsistently, and two runs that
differ in a step nobody wrote down are two experiments sharing a `run_id`. So
everything R1 does is a subcommand here, and every subcommand prints the
`run_id` it produced or consumed.

    orch-workers sweep --tasks data/tasks/mbpp.jsonl --backend vllm_offline \\
        --small Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --large Qwen/Qwen2.5-Coder-7B-Instruct

    orch-workers characterize --backend vllm_openai --hardware A100-80GB \\
        --usd-per-gpu-hour 1.10

    orch-workers impute --run 2026-08-14-a3f91c2-7d4e08
    orch-workers validate --run 2026-08-14-a3f91c2-7d4e08
    orch-workers runs
    orch-workers show --run 2026-08-14-a3f91c2-7d4e08
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .arms import ARMS, DEFAULT_SEEDS, FROZEN_LADDER
from .backends import available, default_backend_name, get_backend
from .cost import CostCoefficients, DEFAULT_COEFFICIENTS_PATH, validate_imputation
from .errors import WorkerError
from .extract import extract
from .store import list_runs, read_generations, read_manifest
from .sweep import DEFAULT_OUT_ROOT, SweepConfig, run_sweep


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", default=None,
        help=f"generation backend; one of {available()} "
             f"(default: {default_backend_name()})",
    )
    parser.add_argument("--small", default="mock-small",
                        help="model id backing the 'small' rung")
    parser.add_argument("--large", default="mock-large",
                        help="model id backing the 'large' rung")
    parser.add_argument(
        "--backend-option", action="append", default=[], metavar="KEY=VALUE",
        help="extra backend argument, repeatable (e.g. base_url=http://host:8000/v1)",
    )


def _backend_options(pairs: list[str]) -> dict[str, object]:
    """Parse `KEY=VALUE` pairs, coercing the obvious scalar types.

    JSON first, so `--backend-option gpu_memory_utilization=0.9` arrives as a
    float and `enforce_eager=true` as a bool, rather than every option landing
    as a string the backend then has to guess about.
    """
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--backend-option expects KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orch-workers",
        description="R1 — serving and workers: sweeps, characterization, cost.",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug-level logging")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help=f"rollout store root (default: {DEFAULT_OUT_ROOT})")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- sweep ---------------------------------------------------------------
    sweep = sub.add_parser("sweep", help="generate rollouts, resumably")
    sweep.add_argument("--tasks", required=True, type=Path,
                       help="task manifest .jsonl, or a directory of them")
    sweep.add_argument("--splits", type=Path, default=None,
                       help="splits.json from R4")
    sweep.add_argument("--arms", nargs="+", default=list(FROZEN_LADDER),
                       choices=sorted(ARMS), help="arms to generate")
    sweep.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    sweep.add_argument("--include-splits", nargs="+", default=None,
                       help="restrict to these splits (default: all)")
    sweep.add_argument("--limit", type=int, default=None,
                       help="first N tasks in manifest order")
    sweep.add_argument("--batch-size", type=int, default=32,
                       help="requests per backend call, or serving concurrency")
    sweep.add_argument("--dataset", default="", help="dataset label for every row")
    sweep.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS_PATH)
    sweep.add_argument("--no-resume", action="store_true",
                       help="ignore existing rows (they are still not overwritten)")
    sweep.add_argument("--no-retry-errors", action="store_true",
                       help="treat previously-errored cells as complete")
    sweep.add_argument("--truncation-alarm", type=float, default=0.15)
    _add_backend_args(sweep)

    # -- characterize --------------------------------------------------------
    char = sub.add_parser(
        "characterize",
        help="fit cost/latency coefficients on a serving backend",
    )
    char.add_argument("--out", type=Path, default=DEFAULT_COEFFICIENTS_PATH)
    char.add_argument("--roles", nargs="+", default=["small", "large"],
                      choices=["small", "large"],
                      help="rungs to measure. One server hosts one model, so "
                           "characterize the rung whose weights are actually "
                           "resident; the others are carried over from --out")
    char.add_argument("--concurrency", nargs="+", type=int, default=[1, 8],
                      help="declared in-flight request counts to measure")
    char.add_argument("--hardware", default="unspecified",
                      help="what this was measured on; recorded, not inferred")
    char.add_argument("--usd-per-gpu-hour", type=float, default=0.0,
                      help="instance rate; recorded as a stated assumption")
    char.add_argument("--repeats", type=int, default=3,
                      help="samples per grid cell")
    char.add_argument("--threshold", type=float, default=0.9,
                      help="R^2 the imputation must clear")
    char.add_argument("--allow-approx-tokens", action="store_true",
                      help="permit estimated token counts (never for reported numbers)")
    char.add_argument("--notes", default="")
    _add_backend_args(char)

    # -- impute --------------------------------------------------------------
    imp = sub.add_parser(
        "impute", help="write a cost sidecar for a run, leaving its rows untouched")
    imp.add_argument("--run", required=True, help="run_id")
    imp.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS_PATH)
    imp.add_argument("--overwrite", action="store_true")
    imp.add_argument("--allow-unsealed", action="store_true")

    # -- validate ------------------------------------------------------------
    val = sub.add_parser(
        "validate", help="check imputed latency against measured wall-clock")
    val.add_argument("--run", required=True, help="run_id")
    val.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS_PATH)
    val.add_argument("--threshold", type=float, default=0.9)

    # -- runs / show ---------------------------------------------------------
    runs = sub.add_parser("runs", help="list runs in the store")
    runs.add_argument("--all", action="store_true",
                      help="include unsealed runs, which readers normally skip")

    show = sub.add_parser("show", help="print a run's manifest summary")
    show.add_argument("--run", required=True, help="run_id")
    show.add_argument("--json", action="store_true", help="print the raw manifest")

    # -- extract -------------------------------------------------------------
    ext = sub.add_parser(
        "extract", help="run the code extractor on a file, for debugging prompts")
    ext.add_argument("path", type=Path, help="file containing a model response")
    ext.add_argument("--entrypoint", default="", help="symbol the tests import")

    return parser


# -- commands ----------------------------------------------------------------


def _cmd_sweep(args: argparse.Namespace) -> int:
    config = SweepConfig(
        tasks_path=args.tasks,
        splits_path=args.splits,
        out_root=args.runs_root,
        arms=tuple(args.arms),
        seeds=tuple(args.seeds),
        include_splits=tuple(args.include_splits) if args.include_splits else None,
        limit=args.limit,
        backend=args.backend,
        small_model=args.small,
        large_model=args.large,
        backend_options=_backend_options(args.backend_option),
        batch_size=args.batch_size,
        coefficients_path=args.coefficients,
        resume=not args.no_resume,
        retry_errors=not args.no_retry_errors,
        truncation_alarm=args.truncation_alarm,
        dataset=args.dataset,
    )
    report = run_sweep(config)
    print(report.summary())
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"run_id: {report.run_id}")
    # A truncation alarm is a failed sweep, not a note. Exiting non-zero is
    # what stops it being scripted past.
    return 1 if report.truncation_rate > config.truncation_alarm else 0


def _cmd_characterize(args: argparse.Namespace) -> int:
    from .characterize import build_probes, run_and_save

    backend = get_backend(
        args.backend, small_model=args.small, large_model=args.large,
        **_backend_options(args.backend_option),
    )
    try:
        result = run_and_save(
            backend, args.out,
            roles=tuple(args.roles),
            concurrencies=tuple(args.concurrency),
            probes=build_probes(repeats=args.repeats),
            hardware=args.hardware,
            usd_per_gpu_hour=args.usd_per_gpu_hour,
            require_exact_tokens=not args.allow_approx_tokens,
            threshold=args.threshold,
            notes=args.notes,
        )
    finally:
        backend.close()

    print(result.report())
    print(f"wrote {args.out}")
    if not result.passed:
        print(
            "\nimputed latency does not track measured wall-clock at the "
            "required R^2. The imputation is not trustworthy yet — say so "
            "before anyone builds a cost-accuracy frontier on it.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_impute(args: argparse.Namespace) -> int:
    from .impute import impute_run

    coefficients = CostCoefficients.load(args.coefficients)
    report = impute_run(
        args.runs_root, args.run, coefficients,
        allow_unsealed=args.allow_unsealed, overwrite=args.overwrite,
    )
    print(report.summary())
    print(f"wrote {report.path}")
    return 1 if report.n_unpriced else 0


def _cmd_validate(args: argparse.Namespace) -> int:
    coefficients = CostCoefficients.load(args.coefficients)
    reports = validate_imputation(
        read_generations(args.runs_root, args.run),
        coefficients, threshold=args.threshold,
        batch=coefficients.reference_batch,
    )
    if not reports:
        print(
            f"no eligible rows in {args.run}: validation needs serving-mode rows "
            f"with exact token counts at batch={coefficients.reference_batch}. "
            f"A batched sweep's wall-clock measures queue depth and cannot "
            f"validate a latency model.",
            file=sys.stderr,
        )
        return 1
    for report in reports:
        print(report.summary())
    return 0 if all(r.passed for r in reports) else 1


def _cmd_runs(args: argparse.Namespace) -> int:
    runs = list_runs(args.runs_root, sealed_only=not args.all)
    if not runs:
        print(f"no runs under {args.runs_root}")
        return 0
    for run_id in runs:
        try:
            manifest = read_manifest(args.runs_root, run_id)
            flag = "" if manifest.get("publishable") else "  [not publishable]"
            print(f"{run_id}  {manifest.get('n_rows', 0):>7} rows{flag}")
        except WorkerError:
            print(f"{run_id}  {'':>7}       [unsealed]")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.runs_root, args.run)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    print(f"run_id        {manifest['run_id']}")
    print(f"sealed_at     {manifest['sealed_at']}")
    print(f"rows          {manifest['n_rows']}")
    print(f"publishable   {manifest['publishable']}")
    print(f"truncation    {manifest['truncation_rate']:.2%}")
    print(f"schema        v{manifest['schema_version']}")
    for label, key in (("by arm", "by_arm"),
                       ("finish reason", "by_finish_reason"),
                       ("extraction", "by_extract_strategy")):
        print(f"\n{label}:")
        for name, count in (manifest["counts"].get(key) or {}).items():
            print(f"  {name or '(none)':<24} {count:>7}")
    for arm, hashes in (manifest.get("params_hash_by_arm") or {}).items():
        if len(hashes) > 1:
            print(f"\nWARNING: arm {arm} holds multiple params_hash values: {hashes}")
    for warning in (manifest.get("extra", {}).get("warnings") or []):
        print(f"\nwarning: {warning}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    result = extract(args.path.read_text(encoding="utf-8"), entrypoint=args.entrypoint)
    print(f"strategy   {result.strategy}")
    print(f"blocks     {result.n_blocks}")
    print(f"parses     {result.parses}")
    print(f"entrypoint {result.defines_entrypoint}")
    print("-" * 60)
    print(result.code)
    return 0 if result.parses else 1


_COMMANDS = {
    "sweep": _cmd_sweep,
    "characterize": _cmd_characterize,
    "impute": _cmd_impute,
    "validate": _cmd_validate,
    "runs": _cmd_runs,
    "show": _cmd_show,
    "extract": _cmd_extract,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return _COMMANDS[args.command](args)
    except WorkerError as exc:
        # Expected, actionable failures print their message. A traceback here
        # would bury a message that was written to be read.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; run the same command again to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
