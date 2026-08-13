"""`orch-eval` — the command line R4 actually runs.

Four subcommands, matching the four things R4 does:

    report   build results.json from a pinned run_id
    audit    run the leakage audit, exit non-zero if blocked
    power    how many tasks are needed, from measured discordance
    golden   diff a fresh report against the committed golden file

Every subcommand requires an explicit `--run-id`. Nothing reads "latest": a
report whose inputs depend on directory mtime is not reproducible, and the
failure is silent because the number still looks fine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .leakage import audit as run_audit
from .loading import StoreError, load_run, unlock_test_split
from .policies import standard_baselines
from .report import build, compare_to_golden, format_table
from .stats import mcnemar_sample_size

DEFAULT_LAMS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="runs", help="rollout store root")
    parser.add_argument("--run-id", required=True,
                        help="exact run to read; nothing reads 'latest'")
    parser.add_argument("--splits", default="train,val",
                        help="comma-separated. 'test' requires --unlock-prereg")
    parser.add_argument(
        "--unlock-prereg", default=None,
        help="path to the pre-registered analysis; required to read the test "
             "split, and the file must already exist",
    )
    parser.add_argument("--allow-dirty", action="store_true",
                        help="read a -dirty run (local iteration only)")


def _load(args: argparse.Namespace):
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    unlock = None
    if "test" in splits:
        if not args.unlock_prereg:
            raise SystemExit(
                "reading the test split requires --unlock-prereg pointing at "
                "the pre-registered analysis. Write down the metric, "
                "comparison, test, correction, and stopping rule first."
            )
        unlock = unlock_test_split(
            reason="cli", preregistration=args.unlock_prereg
        )
    return load_run(
        args.root, args.run_id, splits=splits, unlock=unlock,
        allow_dirty=args.allow_dirty,
    )


def cmd_report(args: argparse.Namespace) -> int:
    store = _load(args)
    report = build(
        store, standard_baselines(), lams=DEFAULT_LAMS,
        n_resamples=args.resamples, seed=args.seed,
    )
    print(format_table(report))
    if args.out:
        path = report.write(args.out)
        print(f"\nwrote {path}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    store = _load(args)
    report = run_audit(store, reported_split=args.reported_split)
    print(report)
    if report.blocked:
        print("\nAUDIT BLOCKED — not publishable", file=sys.stderr)
        return 2
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    n = mcnemar_sample_size(
        discordant_rate=args.discordance,
        odds_ratio=args.odds_ratio,
        alpha=args.alpha,
        power=args.power,
    )
    print(
        f"tasks needed: {n}\n"
        f"  discordance {args.discordance:.3f}  odds ratio {args.odds_ratio:.3f}  "
        f"alpha {args.alpha}  power {args.power}\n"
        f"\nNote this scales with the DISCORDANT rate, not the accuracy gap. "
        f"Two arms that agree on almost everything need a far larger corpus "
        f"than their gap suggests — measure it on the pilot, do not guess."
    )
    return 0


def cmd_golden(args: argparse.Namespace) -> int:
    store = _load(args)
    report = build(
        store, standard_baselines(), lams=DEFAULT_LAMS,
        n_resamples=args.resamples, seed=args.seed,
    )
    diffs = compare_to_golden(report, args.golden)
    if not diffs:
        print(f"golden run reproduces exactly ({args.golden})")
        return 0
    print(f"{len(diffs)} differences against {args.golden}:", file=sys.stderr)
    for path in diffs[:40]:
        print(f"  {path}", file=sys.stderr)
    if len(diffs) > 40:
        print(f"  ... and {len(diffs) - 40} more", file=sys.stderr)
    print(
        "\nIf this change is intended, mark the PR body BREAKING-GOLDEN: and "
        "explain the diff. If it is not, a statistic moved without anyone "
        "meaning it to.",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orch-eval", description="R4 evaluation harness"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="build results.json")
    _add_store_args(p_report)
    p_report.add_argument("--out", default=None, help="directory to seal into")
    p_report.add_argument("--resamples", type=int, default=10_000)
    p_report.add_argument("--seed", type=int, default=0)
    p_report.set_defaults(func=cmd_report)

    p_audit = sub.add_parser("audit", help="run the leakage audit")
    _add_store_args(p_audit)
    p_audit.add_argument("--reported-split", default="test")
    p_audit.set_defaults(func=cmd_audit)

    p_power = sub.add_parser("power", help="tasks needed for an effect")
    p_power.add_argument("--discordance", type=float, required=True,
                         help="measured discordant rate from the pilot")
    p_power.add_argument("--odds-ratio", type=float, default=1.45)
    p_power.add_argument("--alpha", type=float, default=0.05)
    p_power.add_argument("--power", type=float, default=0.80)
    p_power.set_defaults(func=cmd_power)

    p_golden = sub.add_parser("golden", help="diff against the golden report")
    _add_store_args(p_golden)
    p_golden.add_argument("--golden", required=True, help="committed results.json")
    p_golden.add_argument("--resamples", type=int, default=10_000)
    p_golden.add_argument("--seed", type=int, default=0)
    p_golden.set_defaults(func=cmd_golden)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (StoreError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
