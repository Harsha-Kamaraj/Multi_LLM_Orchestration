"""`orch-eval` — the command line R4 actually runs.

Seven subcommands, matching the seven things R4 does:

    report   build results.json from a pinned run_id
    audit    run the leakage audit, exit non-zero if blocked
    power    how many tasks are needed, from measured discordance
    golden   diff a fresh report against the committed golden file
    splits   build or verify the frozen train/val/test manifest
    taxonomy why generations failed, attributed per arm
    gate     adjudicate the four Phase 0 gate quantities

Every store-reading subcommand requires an explicit `--run-id`. Nothing reads
"latest": a report whose inputs depend on directory mtime is not reproducible,
and the failure is silent because the number still looks fine.
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
from .splits import (
    SplitError,
    SplitManifest,
    build as build_splits,
    read_corpus,
    verify as verify_splits,
)
from .gates import evaluate as evaluate_gates
from .stats import mcnemar_sample_size
from .taxonomy import classify_store, explain_gap

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
        print(f"\nwrote {path}\nwrote {path.parent / 'report.html'}")
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


def cmd_splits(args: argparse.Namespace) -> int:
    tasks = read_corpus(args.corpus)
    if args.verify:
        manifest = SplitManifest.read(args.verify)
        problems = verify_splits(manifest, tasks)
        if problems:
            print(f"{len(problems)} problems with {args.verify}:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(f"{args.verify} verifies against {len(tasks)} tasks")
        return 0

    manifest = build_splits(tasks, name=args.name, salt=args.salt)
    print(f"corpus      {len(tasks)} tasks   hash {manifest.corpus_hash[:16]}")
    for split, count in manifest.counts.items():
        print(f"  {split:<6} {count:>6}  {count / manifest.n_tasks:>7.1%}")
    if args.out:
        print(f"wrote {manifest.write(args.out)}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    store = _load(args)
    corpus = read_corpus(args.corpus) if args.corpus else None
    report = evaluate_gates(store, corpus=corpus)
    print(report)
    if report.blocked:
        print("\nPHASE 0 BLOCKED — a hard-stop gate failed", file=sys.stderr)
        return 2
    if not report.all_measured or report.failures:
        return 1
    return 0


def cmd_taxonomy(args: argparse.Namespace) -> int:
    store = _load(args)
    taxonomy = classify_store(store)
    print(taxonomy)
    if (dominant := taxonomy.dominant_failure()):
        from .taxonomy import MEANING

        print(f"\ndominant failure: {dominant} — {MEANING[dominant]}")
    if (lines := explain_gap(taxonomy)):
        print("\ngap attribution:")
        for line in lines:
            print(f"  {line}")
    return 0


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

    p_gate = sub.add_parser("gate", help="adjudicate the Phase 0 gate")
    _add_store_args(p_gate)
    p_gate.add_argument("--corpus", default=None,
                        help="R2 task manifest; required to measure AUC_D0, "
                             "whose features are not on the rollout row")
    p_gate.set_defaults(func=cmd_gate)

    p_tax = sub.add_parser("taxonomy", help="why generations failed")
    _add_store_args(p_tax)
    p_tax.set_defaults(func=cmd_taxonomy)

    p_splits = sub.add_parser("splits", help="build or verify frozen splits")
    p_splits.add_argument("--corpus", required=True, help="R2 task manifest (jsonl)")
    p_splits.add_argument("--name", default="pilot")
    p_splits.add_argument(
        "--salt", default="pilot-2026-08-13",
        help="the split's identity. Re-salting after seeing results is the "
             "split-level equivalent of unblinding twice",
    )
    p_splits.add_argument("--out", default=None, help="path to write the manifest")
    p_splits.add_argument("--verify", default=None,
                          help="existing manifest to check instead of building")
    p_splits.set_defaults(func=cmd_splits)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (StoreError, SplitError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
