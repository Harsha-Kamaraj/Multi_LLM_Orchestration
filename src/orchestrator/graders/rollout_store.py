"""Append-only store for graded rollout rows.

Mirrors R1's `workers/store.py` write-then-seal / checksummed pattern
(append-only, one part file per writing process, a manifest whose existence
is the seal), but lives on this side of the boundary and never touches R1's
files: R1 seals `runs/{run_id}/generations/` — ungraded, immutable — and this
module seals `runs/{run_id}/rollouts/`, the same rows with grading fields
filled in.

**"Mutate a graded row; a re-grade is a new run_id"** (diya.md). A sealed
rollouts store refuses further appends, exactly like R1's generations store
refuses appends to a sealed run — grading again means grading a new run_id,
never overwriting this one.

```
runs/{run_id}/
    _CONFIG.json                  R1's — untouched
    generations/part-*.jsonl      R1's — untouched
    _MANIFEST.json                R1's seal marker — untouched
    rollouts/part-*.jsonl         ours — append-only
    rollouts/rollouts.parquet     ours — derived, optional (needs pyarrow)
    _ROLLOUT_MANIFEST.json        ours — write-then-seal, distinct name so
                                   it can never collide with R1's manifest
                                   in the same run_id directory
```
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import GraderError

MANIFEST_NAME = "_ROLLOUT_MANIFEST.json"
ROWS_DIR = "rollouts"
PARQUET_NAME = "rollouts.parquet"

_FSYNC_EVERY = 50


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RolloutStore:
    """Writer and reader for one run's graded rollout rows."""

    def __init__(self, root: Path | str, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.dir = self.root / run_id
        self.rows_dir = self.dir / ROWS_DIR
        self._fh: Any = None
        self._part: Path | None = None
        self._since_sync = 0

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST_NAME

    @property
    def is_sealed(self) -> bool:
        return self.manifest_path.exists()

    def part_files(self) -> list[Path]:
        if not self.rows_dir.exists():
            return []
        return sorted(self.rows_dir.glob("part-*.jsonl"))

    # -- writing ---------------------------------------------------------

    def open(self) -> "RolloutStore":
        if self.is_sealed:
            raise GraderError(
                f"run {self.run_id} already has a sealed rollouts store; "
                f"a re-grade is a new run_id, never an overwrite"
            )
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        shard = f"{os.getpid():06d}-{int(time.time()):010d}-{os.urandom(3).hex()}"
        self._part = self.rows_dir / f"part-{shard}.jsonl"
        self._fh = self._part.open("a", encoding="utf-8", newline="")
        return self

    def append(self, row: dict[str, Any]) -> None:
        """Append one graded row. Never rewrites, never reorders.

        `row` must carry `rollout_id` — the identity a re-grade under a new
        `run_id` is still traceable by.
        """
        if self._fh is None:
            raise GraderError("store is not open; call open() before append()")
        if "rollout_id" not in row:
            raise GraderError("graded row is missing rollout_id")
        self._fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        self._since_sync += 1
        if self._since_sync >= _FSYNC_EVERY:
            self.flush()

    def append_many(self, rows: Iterable[dict[str, Any]]) -> int:
        n = 0
        for row in rows:
            self.append(row)
            n += 1
        return n

    def flush(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._since_sync = 0

    def close(self) -> None:
        if self._fh is not None:
            self.flush()
            self._fh.close()
            self._fh = None

    # -- sealing -----------------------------------------------------------

    def seal(self, *, extra: dict[str, Any] | None = None,
             write_parquet: bool = True) -> dict[str, Any]:
        """Write `_ROLLOUT_MANIFEST.json`, making the graded run valid to read.

        Counts are recomputed from the files on disk, not this process's
        counters — a re-grade resumed by a second process must describe every
        row on disk, not just the ones this process wrote.
        """
        self.close()
        if self.is_sealed:
            raise GraderError(f"run {self.run_id} rollouts are already sealed")

        parts = self.part_files()
        files: list[dict[str, Any]] = []
        stats: Counter[str] = Counter()
        total = 0

        for part in parts:
            rows = 0
            for row in _read_jsonl(part):
                rows += 1
                stats[f"error_class:{row.get('error_class')}"] += 1
                for flag in row.get("hack_flags") or []:
                    stats[f"hack:{flag}"] += 1
                if row.get("hidden_total") and row.get("hidden_passed") == row.get("hidden_total"):
                    stats["solved"] += 1
            total += rows
            files.append({
                "name": part.name,
                "rows": rows,
                "bytes": part.stat().st_size,
                "sha256": _sha256_file(part),
            })

        parquet_written = False
        if write_parquet and total:
            parquet_written = self._write_parquet(parts)

        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "sealed_at": _utc_now(),
            "n_rows": total,
            "files": files,
            "parquet": PARQUET_NAME if parquet_written else None,
            "solved_count": stats.get("solved", 0),
            "counts": {
                "by_error_class": {k[len("error_class:"):]: v for k, v in sorted(stats.items())
                                    if k.startswith("error_class:")},
                "by_hack_flag": {k[len("hack:"):]: v for k, v in sorted(stats.items())
                                  if k.startswith("hack:")},
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }
        if extra:
            manifest["extra"] = extra

        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8", newline="",
        )
        tmp.replace(self.manifest_path)
        return manifest

    def _write_parquet(self, parts: list[Path]) -> bool:
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError:
            return False
        rows = [row for part in parts for row in _read_jsonl(part)]
        if not rows:
            return False
        # `extra` (passed through from R1's Generation row) holds
        # backend-specific keys whose shape varies between rows, and is
        # often `{}` — Arrow infers a struct type from the first row and
        # then fails on every other row, exactly as R1's own store.py notes.
        # Serialize it so the column type is stable regardless of content.
        for row in rows:
            if isinstance(row.get("extra"), dict):
                row["extra"] = json.dumps(row["extra"], default=str, sort_keys=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, self.rows_dir / PARQUET_NAME, compression="zstd")
        return True

    def __enter__(self) -> "RolloutStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_manifest(root: Path | str, run_id: str) -> dict[str, Any]:
    path = Path(root) / run_id / MANIFEST_NAME
    if not path.exists():
        raise GraderError(
            f"run {run_id} has no {MANIFEST_NAME} and is not valid to read; "
            f"it has not been graded yet or grading was interrupted"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(root: Path | str, run_id: str, *,
              allow_unsealed: bool = False) -> Iterator[dict[str, Any]]:
    store = RolloutStore(root, run_id)
    if not store.dir.exists():
        raise GraderError(f"no such run: {store.dir}")
    if not store.is_sealed and not allow_unsealed:
        raise GraderError(
            f"run {run_id} rollouts are not sealed; readers skip unsealed "
            f"grading runs. Pass allow_unsealed=True only if resuming."
        )
    for part in store.part_files():
        yield from _read_jsonl(part)


def list_graded_runs(root: Path | str, *, sealed_only: bool = True) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if sealed_only and not (child / MANIFEST_NAME).exists():
            continue
        out.append(child.name)
    return out
