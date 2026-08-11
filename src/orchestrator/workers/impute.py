"""Backfilling imputed cost onto a sealed run, without touching a single row.

A sweep can finish before any characterization exists — that is the whole point
of imputing cost from stored token counts rather than measuring it during
generation. But a sealed run is immutable: its manifest carries per-file
checksums, and rewriting rows to add two columns would invalidate every number
already computed from it.

So imputed cost lands **beside** the rows, in a sidecar keyed by a fingerprint
of the coefficients that produced it:

```
runs/{run_id}/cost/{fingerprint}.jsonl      rollout_id -> gpu_seconds, latency, usd
runs/{run_id}/cost/{fingerprint}.json       which coefficients, and their fit quality
```

This is what makes "re-characterize on new hardware and the rollouts do not
need regenerating" true rather than aspirational. Two fingerprints can coexist
for one run — the A100 costing and the H100 costing of the same generations —
and a consumer pins the one it means, exactly as it pins a `run_id`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .cost import CostCoefficients
from .errors import StoreError
from .store import RolloutStore, read_rows

COST_DIR = "cost"


def fingerprint(coefficients: CostCoefficients) -> str:
    """Eight hex characters identifying one set of coefficients.

    Covers the fitted numbers and the dollar rate, but not `created_at` or
    `notes` — re-running an identical characterization should reuse the same
    sidecar rather than producing a second copy under a new name.
    """
    payload = {
        "reference_batch": coefficients.reference_batch,
        "usd_per_gpu_hour": coefficients.usd_per_gpu_hour,
        "hardware": coefficients.hardware,
        "models": {
            model: {
                str(batch): [
                    fit.intercept_s, fit.prefill_s_per_token, fit.decode_s_per_token,
                ]
                for batch, fit in sorted(by_batch.items())
            }
            for model, by_batch in sorted(coefficients.models.items())
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


@dataclass
class ImputeReport:
    """What an imputation pass covered, and what it could not."""

    run_id: str
    fingerprint: str
    path: Path
    n_rows: int = 0
    n_imputed: int = 0
    n_unpriced: int = 0
    total_gpu_seconds: float = 0.0
    total_usd: float = 0.0
    unpriced_models: set[str] = field(default_factory=set)

    @property
    def coverage(self) -> float:
        return self.n_imputed / self.n_rows if self.n_rows else 0.0

    def summary(self) -> str:
        line = (
            f"imputed {self.n_imputed}/{self.n_rows} rows "
            f"({self.coverage:.1%}) for run {self.run_id} "
            f"under coefficients {self.fingerprint}: "
            f"{self.total_gpu_seconds:.1f} GPU-seconds, ${self.total_usd:.4f}"
        )
        if self.unpriced_models:
            line += (
                f"\n  {self.n_unpriced} rows left unpriced — no coefficients for "
                f"{sorted(self.unpriced_models)}. Characterize these models "
                f"before any cost number from this run is reported."
            )
        return line


def cost_dir(root: Path | str, run_id: str) -> Path:
    return Path(root) / run_id / COST_DIR


def impute_run(root: Path | str, run_id: str, coefficients: CostCoefficients,
               *, allow_unsealed: bool = False,
               overwrite: bool = False) -> ImputeReport:
    """Write a cost sidecar for every row of a run.

    Refuses to overwrite an existing sidecar unless asked. Two identical
    coefficient sets produce the same fingerprint, so a repeated pass is
    normally a no-op — and if the file's contents would differ under the same
    fingerprint, something is wrong that silently overwriting would hide.
    """
    store = RolloutStore(root, run_id)
    if not store.exists:
        raise StoreError(f"no such run: {store.dir}")

    fp = fingerprint(coefficients)
    out_dir = cost_dir(root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{fp}.jsonl"
    meta_path = out_dir / f"{fp}.json"

    if rows_path.exists() and not overwrite:
        report = _report_from_existing(run_id, fp, rows_path)
        return report

    report = ImputeReport(run_id=run_id, fingerprint=fp, path=rows_path)
    tmp = rows_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        for row in read_rows(root, run_id, allow_unsealed=allow_unsealed):
            report.n_rows += 1
            model_id = str(row.get("model_id") or "")
            if not coefficients.has(model_id):
                report.n_unpriced += 1
                report.unpriced_models.add(model_id)
                continue
            prefill = int(row.get("prefill_tokens") or 0)
            decode = int(row.get("decode_tokens") or 0)
            gpu_s = coefficients.gpu_seconds(model_id, prefill, decode)
            latency = coefficients.imputed_latency_s(model_id, prefill, decode)
            usd = coefficients.usd(model_id, prefill, decode)
            fh.write(json.dumps({
                "rollout_id": row.get("rollout_id"),
                "task_id": row.get("task_id"),
                "arm": row.get("arm"),
                "seed": row.get("seed"),
                "model_id": model_id,
                "gpu_seconds": gpu_s,
                "imputed_latency_s": latency,
                "usd": usd,
            }) + "\n")
            report.n_imputed += 1
            report.total_gpu_seconds += gpu_s
            report.total_usd += usd
    tmp.replace(rows_path)

    meta_path.write_text(json.dumps({
        "run_id": run_id,
        "fingerprint": fp,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coefficients": coefficients.to_dict(),
        "n_rows": report.n_rows,
        "n_imputed": report.n_imputed,
        "n_unpriced": report.n_unpriced,
        "unpriced_models": sorted(report.unpriced_models),
        "total_gpu_seconds": report.total_gpu_seconds,
        "total_usd": report.total_usd,
    }, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="")
    return report


def _report_from_existing(run_id: str, fp: str, path: Path) -> ImputeReport:
    """Summarize a sidecar that already exists, without rewriting it."""
    report = ImputeReport(run_id=run_id, fingerprint=fp, path=path)
    for entry in read_imputed(path):
        report.n_rows += 1
        report.n_imputed += 1
        report.total_gpu_seconds += float(entry.get("gpu_seconds") or 0.0)
        report.total_usd += float(entry.get("usd") or 0.0)
    return report


def read_imputed(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream a sidecar's entries."""
    path = Path(path)
    if not path.exists():
        raise StoreError(f"no cost sidecar at {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def list_sidecars(root: Path | str, run_id: str) -> list[str]:
    """Fingerprints of every costing available for a run."""
    directory = cost_dir(root, run_id)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.jsonl"))
