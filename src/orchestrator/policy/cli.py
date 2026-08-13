"""`orch-policy` — R3's operational surface.

One command per thing R3 produces, each pinned to an explicit `run_id`. There
is no subcommand that resolves "latest" and no flag that opens the test split;
both absences are the enforcement rather than a check.

    orch-policy gate    --run <run_id> --tasks data/tasks/pilot.jsonl
    orch-policy fixture --out runs --tasks 400
    orch-policy runs

`gate` exits non-zero when `AUC_D1` fails its threshold. That is ROADMAP.md's
hard stop wired to a process exit code, so a scheduled run of it is a tripwire
rather than something a human has to remember to read.

**INCONCLUSIVE also exits non-zero.** An interval straddling the threshold has
not cleared it; treating "cannot tell" as success is how a gate stops being a
gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import fixtures, gate, store
from .errors import PolicyError
from .features import feature_set

DEFAULT_ROOT = Path("runs")

#: Exit codes, so a caller can tell the three outcomes apart without parsing.
EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2
EXIT_INCONCLUSIVE = 3


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="store root (default: runs/)")
    parser.add_argument("--run", required=True,
                        help="run_id, pinned explicitly — nothing reads 'latest'")
    parser.add_argument("--tasks", type=Path, default=None,
                        help="R2's task manifest. Required for D0 features: "
                             "the rollout row carries no prompt")
    parser.add_argument("--cost", default=None, metavar="FINGERPRINT",
                        help="which costing to attach, from `orch-policy runs`")
    parser.add_argument("--layer", choices=("rollouts", "generations"),
                        default=None,
                        help="which layer to read. Defaults to whichever is "
                             "sealed, preferring R2's graded `rollouts` — the "
                             "labels exist nowhere else")


def _cmd_gate(args: argparse.Namespace) -> int:
    data = store.load_rollouts(
        args.root, args.run,
        tasks_path=args.tasks,
        cost_fingerprint=args.cost,
        layer=args.layer,
    )

    points = ("D0", "D1") if args.decision_point == "both" else (args.decision_point,)
    results = {
        point: gate.measure_gate(
            data, point,
            arm=args.arm,
            features=feature_set(point, with_probe=args.with_probe and point == "D1"),
            n_resamples=args.resamples,
            seed=args.seed,
        )
        for point in points
    }

    print(gate.gate_report(results))

    if args.out:
        path = gate.write_gate_report(
            results, args.out,
            cost_fingerprint=args.cost,
            tasks_path=str(args.tasks) if args.tasks else None,
            publishable=data.publishable,
        )
        print(f"\nwrote {path}")

    if not data.publishable:
        print(
            "\nNOTE: this run is not publishable — it was swept from a dirty "
            "worktree, so the recorded git sha does not describe the code that "
            "produced the rows. Fine to develop against, not to report from.",
            file=sys.stderr,
        )

    # The hard stop is on D1. D0 failing is expected and is not fatal: it means
    # pre-generation routing is dead and the work moves to D1.
    verdicts = {point: result.verdict for point, result in results.items()}
    if verdicts.get("D1") == "FAIL":
        return EXIT_GATE_FAILED
    if verdicts.get("D1") == "INCONCLUSIVE":
        return EXIT_INCONCLUSIVE
    return EXIT_OK


def _cmd_fixture(args: argparse.Namespace) -> int:
    from schemas.synth import SynthConfig

    fixture = fixtures.write_fixture(
        args.out,
        SynthConfig(n_tasks=args.tasks, seeds=args.seeds,
                    d0_signal=args.d0_signal, d1_fidelity=args.d1_fidelity),
        seed=args.seed,
        layout=args.layout,
    )
    print(f"run_id     {fixture.run_id}")
    print(f"rows       {len(fixture.result.rows)}")
    print(f"tasks      {fixture.tasks_path}")
    print(f"costing    {fixture.cost_fingerprint}")
    print(f"layout     {args.layout}")
    print(f"\nmeasure it with:\n"
          f"  orch-policy gate --root {args.out} --run {fixture.run_id} "
          f"--tasks {fixture.tasks_path} --cost {fixture.cost_fingerprint}")
    return EXIT_OK


def _cmd_runs(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.exists():
        print(f"no store at {root}")
        return EXIT_OK
    found = False
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / "_MANIFEST.json").exists():
            continue
        found = True
        manifest = json.loads((child / "_MANIFEST.json").read_text())
        costings = store.list_cost_fingerprints(root, child.name)
        flag = "" if manifest.get("publishable", False) else "  [not publishable]"
        graded = "graded" if (child / "_ROLLOUT_MANIFEST.json").exists() \
            else "UNGRADED"
        print(f"{child.name}  rows={manifest.get('n_rows', '?')}  "
              f"{graded}  costings={costings or '-'}{flag}")
    if not found:
        print(f"no sealed runs under {root} — readers skip unsealed ones")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orch-policy",
        description="R3 — policy and learning. Reads a pinned rollout store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gate_cmd = sub.add_parser(
        "gate",
        help="measure AUC_D0 and AUC_D1 with clustered intervals",
        description="Phase 0's gate. Exits non-zero if AUC_D1 fails or cannot "
                    "be decided.",
    )
    _add_run_arguments(gate_cmd)
    gate_cmd.add_argument("--decision-point", choices=("D0", "D1", "both"),
                          default="both")
    gate_cmd.add_argument("--arm", default=None,
                          help="the cheap arm; inferred from measured cost "
                               "when a costing is pinned")
    gate_cmd.add_argument("--with-probe", action="store_true",
                          help="include sibling features at D1. They oblige "
                               "the policy to pay for the draws they read")
    gate_cmd.add_argument("--resamples", type=int, default=2000)
    gate_cmd.add_argument("--seed", type=int, default=0)
    gate_cmd.add_argument("--out", type=Path, default=None,
                          help="write the verdict as JSON")
    gate_cmd.set_defaults(func=_cmd_gate)

    fixture_cmd = sub.add_parser(
        "fixture", help="write a synthetic run with a planted signal")
    fixture_cmd.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    fixture_cmd.add_argument("--tasks", type=int, default=400)
    fixture_cmd.add_argument("--seeds", type=int, default=3)
    fixture_cmd.add_argument("--d0-signal", type=float, default=0.40)
    fixture_cmd.add_argument("--d1-fidelity", type=float, default=0.82)
    fixture_cmd.add_argument("--seed", type=int, default=0)
    fixture_cmd.add_argument("--layout", choices=("generations", "split"),
                             default="generations",
                             help="'split' reproduces the real two-layer shape: "
                                  "R1's ungraded rows plus R2's graded ones")
    fixture_cmd.set_defaults(func=_cmd_fixture)

    runs_cmd = sub.add_parser("runs", help="list sealed runs and their costings")
    runs_cmd.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    runs_cmd.set_defaults(func=_cmd_runs)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except PolicyError as exc:
        # Every refusal in this package explains itself, so the message is the
        # error output — a traceback would bury it.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
