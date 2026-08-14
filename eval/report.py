"""`results.json` — and the byte-identity that makes it a regression gate.

The golden-run test is the single most valuable piece of infrastructure R4
owns. Re-running the pipeline must reproduce `results.json` byte for byte from
a `run_id`. A change that alters it must be marked `BREAKING-GOLDEN:` in the PR
body with the diff explained.

This is what catches a silent statistical bug three weeks after it lands —
someone changes a default, a percentile, or a resampling scheme, every test
still passes because every test was written against the new behaviour, and the
only evidence is that the numbers moved. Byte-identity turns that into a diff.

Byte-identity is not free. Three things are done deliberately for it:

**Every float is rounded at serialization.** Bootstrap output carries noise far
below any digit worth reporting, and unrounded floats differ in the last bits
across numpy versions and platforms. Rounding at the boundary makes the file
stable without pretending the precision exists.

**Every seed is explicit.** A bootstrap without a recorded seed is not
reproducible, and "we ran it again and got roughly the same thing" is not the
claim being made.

**Key order is fixed.** `sort_keys=True` throughout, so a dict iteration-order
change cannot masquerade as a result change.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .confusion import confusion, oracle_headroom
from .frontier import Point, sweep
from .leakage import AuditReport
from .loading import Rollouts
from .policies import Outcome
from .stats import benjamini_hochberg, mcnemar_exact, paired_diff_bootstrap
from .taxonomy import classify_store, explain_gap

# Digits kept in the report. Six is far beyond anything reportable and far
# short of float noise — the window where the file is both stable and honest.
PRECISION = 6

RESULTS_NAME = "results.json"
HTML_NAME = "report.html"
MANIFEST_NAME = "_MANIFEST.json"

# Everything a policy is compared against. Named rather than inferred so a
# report cannot quietly change its reference and show a different win.
REFERENCE_POLICY = "verifier_gated_cascade"


def _round(value: Any) -> Any:
    """Recursively round floats so the serialized file is stable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return None  # JSON has no NaN/Infinity; null is honest about it
        return round(v, PRECISION)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _round(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_round(v) for v in value.tolist()]
    return value


def dumps(payload: Mapping[str, Any]) -> str:
    """Deterministic JSON. The only serializer the report uses."""
    return json.dumps(_round(payload), sort_keys=True, indent=2,
                      ensure_ascii=True) + "\n"


@dataclass
class Report:
    """A complete evaluation result, ready to seal."""

    run_id: str
    payload: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return dumps(self.payload)

    def write(self, directory: str | Path) -> Path:
        """Write-then-seal. The directory is invalid until the manifest lands.

        Readers skip unsealed directories, so a report killed midway is never
        mistaken for a complete one — its numbers look fine, because the ones
        that exist are fine.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        results = out / RESULTS_NAME
        results.write_text(self.to_json(), encoding="utf-8")

        # The human-facing half, rendered from the same payload so the two can
        # never disagree. Imported here rather than at module scope because
        # `html` imports `report` for its type.
        from .html import render

        (out / HTML_NAME).write_text(render(self), encoding="utf-8")

        manifest = {
            "run_id": self.run_id,
            "results": RESULTS_NAME,
            "html": HTML_NAME,
            "sha256": _sha256(results),
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "numpy": np.__version__,
            "platform": platform.system(),
            "precision": PRECISION,
            "warnings": self.warnings,
        }
        (out / MANIFEST_NAME).write_text(dumps(manifest), encoding="utf-8")
        return results


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    store: Rollouts,
    policies: Mapping[str, Callable[[Rollouts], Outcome]],
    *,
    lams: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    reference: str = REFERENCE_POLICY,
    audit: AuditReport | None = None,
    n_resamples: int = 10_000,
    seed: int = 0,
    q: float = 0.05,
) -> Report:
    """Assemble the full comparison into a serializable result.

    Every comparison is against `reference` — the verifier-gated cascade —
    because that is the baseline the project's claim has to survive. A report
    that compares only against `always_small` and `always_large` is answering
    an easier question than the one asked.
    """
    if reference not in policies:
        raise KeyError(
            f"reference policy {reference!r} is not in the comparison set. "
            f"Every claim is measured against it; omitting it would answer an "
            f"easier question. Have: {sorted(policies)}"
        )

    outcomes = {name: fn(store) for name, fn in policies.items()}
    ref = outcomes[reference]

    # --- per-policy summaries ----------------------------------------------
    summaries: dict[str, Any] = {}
    for name, outcome in sorted(outcomes.items()):
        s = outcome.summary()
        conf = confusion(outcome, store)
        summaries[name] = {
            **s,
            "n_replicates": int(outcome.solved.shape[1]),
            "consumes_all_seeds": bool(outcome.consumes_all_seeds),
            "confusion": conf.as_dict(),
            "confusion_warnings": conf.warnings,
        }

    # --- comparisons against the reference ---------------------------------
    comparisons: dict[str, Any] = {}
    p_values: list[float] = []
    names: list[str] = []
    for name, outcome in sorted(outcomes.items()):
        if name == reference:
            continue
        a, b = outcome, ref
        if a.solved.shape != b.solved.shape:
            a, b = a.per_task(), b.per_task()
        ci = paired_diff_bootstrap(
            a.solved, b.solved, n_resamples=n_resamples, seed=seed
        )
        mc = mcnemar_exact(
            a.solved[:, 0].astype(bool), b.solved[:, 0].astype(bool)
        )
        comparisons[name] = {
            "accuracy_difference": ci.as_dict(),
            "mcnemar": {
                "b": mc.b, "c": mc.c, "n_discordant": mc.n_discordant,
                "p_value": mc.p_value, "odds_ratio": mc.odds_ratio,
            },
        }
        p_values.append(mc.p_value)
        names.append(name)

    # --- multiplicity -------------------------------------------------------
    if p_values:
        rejected, adjusted = benjamini_hochberg(p_values, q=q)
        for name, rej, adj in zip(names, rejected, adjusted):
            comparisons[name]["mcnemar"]["p_adjusted"] = float(adj)
            comparisons[name]["mcnemar"]["significant_after_bh"] = bool(rej)

    # --- frontier -----------------------------------------------------------
    frontier = {
        name: [p.as_dict() for p in points]
        for name, points in sweep(policies, store, lams).items()
    }

    # Why generations fail, not just how often. A 20-point gap made of timeouts
    # is a serving problem; the same gap made of wrong answers is a capability
    # problem; made of reward hacks the gap is not real at all.
    taxonomy = classify_store(store)

    warnings = list(store.warnings)
    for name, summary in summaries.items():
        warnings.extend(f"{name}: {w}" for w in summary["confusion_warnings"])

    payload: dict[str, Any] = {
        "run_id": store.run_id,
        "splits": list(store.splits),
        "arms": list(store.arms),
        "n_tasks": store.n_tasks,
        "n_seeds": store.n_seeds,
        "reference_policy": reference,
        "lambda_grid": list(lams),
        "bootstrap": {"n_resamples": n_resamples, "seed": seed, "level": 0.95},
        "multiplicity": {"method": "benjamini_hochberg", "q": q},
        "headroom": oracle_headroom(store),
        "failure_taxonomy": taxonomy.as_dict(),
        "gap_attribution": explain_gap(taxonomy),
        "policies": summaries,
        "comparisons": comparisons,
        "frontier": frontier,
        "warnings": warnings,
    }
    if audit is not None:
        payload["leakage_audit"] = audit.as_dict()
        if audit.blocked:
            warnings.insert(0, "LEAKAGE AUDIT BLOCKED — this report is not publishable")

    return Report(run_id=store.run_id, payload=payload, warnings=warnings)


def compare_to_golden(report: Report, golden: str | Path) -> list[str]:
    """Diff a report against a committed golden file.

    Returns the differing JSON paths, empty when byte-identical. Paths rather
    than a text diff because the useful question is always *which number moved*,
    and a text diff of a 300-line JSON buries it.
    """
    golden_path = Path(golden)
    if not golden_path.exists():
        raise FileNotFoundError(
            f"no golden results at {golden_path}. Generate one and commit it, "
            f"or the regression gate does not exist."
        )
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    actual = json.loads(report.to_json())
    return sorted(_diff_paths(expected, actual))


def _diff_paths(a: Any, b: Any, path: str = "") -> list[str]:
    if type(a) is not type(b):
        return [f"{path or '<root>'} (type {type(a).__name__} -> {type(b).__name__})"]
    if isinstance(a, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            child = f"{path}.{key}" if path else str(key)
            if key not in a:
                out.append(f"{child} (added)")
            elif key not in b:
                out.append(f"{child} (removed)")
            else:
                out.extend(_diff_paths(a[key], b[key], child))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} -> {len(b)})"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_diff_paths(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [f"{path} ({a!r} -> {b!r})"]


def format_table(report: Report) -> str:
    """A human-readable summary. Never a bare mean — every row carries its
    interval, which is a merge blocker in CONTRIBUTING.md."""
    payload = report.payload
    ref = payload["reference_policy"]
    lines = [
        f"run_id: {payload['run_id']}   splits: {','.join(payload['splits'])}   "
        f"tasks: {payload['n_tasks']}",
        f"reference: {ref}",
        "",
        f"{'policy':<26}{'accuracy':>10}{'cost':>9}{'p95 lat':>10}"
        f"{'Δacc vs ref [95% CI]':>30}",
    ]
    for name in sorted(payload["policies"]):
        s = payload["policies"][name]
        if name == ref:
            delta = "— reference —"
        else:
            ci = payload["comparisons"][name]["accuracy_difference"]
            star = "*" if ci["excludes_zero"] else " "
            delta = f"{ci['point']:+.4f} [{ci['low']:+.4f},{ci['high']:+.4f}]{star}"
        lines.append(
            f"{name:<26}{s['accuracy']:>10.4f}{s['cost']:>9.3f}"
            f"{s['latency_p95']:>10.3f}{delta:>30}"
        )
    if payload["warnings"]:
        lines += ["", "warnings:"]
        lines += [f"  - {w}" for w in payload["warnings"]]
    return "\n".join(lines)
