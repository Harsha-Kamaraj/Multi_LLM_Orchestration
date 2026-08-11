"""The rollout store — append-only, partitioned by `run_id`, sealed on close.

```
runs/{run_id}/
    _CONFIG.json                  written when the run opens
    generations/part-{shard}.jsonl    append-only, never rewritten
    generations/generations.parquet   written at seal, when pyarrow is present
    _MANIFEST.json                write-then-seal; its arrival makes the run valid
```

Three properties, each enforced in code rather than by convention:

**Append-only.** Rows are appended and never rewritten. A re-run under changed
conditions is a new `run_id`, so the store accumulates experiments instead of
overwriting them, and any published number can be traced to the exact rows
behind it.

**Write-then-seal.** A run is invalid until `_MANIFEST.json` lands. Readers skip
unsealed runs by default, which is what stops an interrupted sweep from being
read as a complete one — the failure that produces a real number computed over
two thirds of a corpus.

**Checksummed.** The manifest records a sha256 and a row count per part file,
so R4's golden-run test can detect a store that changed underneath a result.

JSONL is authoritative and Parquet is derived. Parquet is what R3 and R4 read,
but it needs pyarrow, and a sweep must not fail at hour nine because an
optional dependency is missing on the GPU box.
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

from .errors import StoreError, UnsealedRunError
from .generation import SCHEMA_VERSION, Generation
from .runid import git_sha7, is_dirty, is_publishable

MANIFEST_NAME = "_MANIFEST.json"
CONFIG_NAME = "_CONFIG.json"
ROWS_DIR = "generations"
PARQUET_NAME = "generations.parquet"

# Rows written between fsyncs. Flushing every row is correct and slow; never
# syncing risks losing the tail of a long sweep to a machine that dies. Fifty
# rows is a few seconds of generation — a cheap amount to lose, and resume
# recovers it.
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
    """Writer and reader for one run's generations.

    Each writing process gets its own part file, keyed by pid and start time.
    Two sweeps that resume the same run therefore never interleave writes into
    one file, which removes the need for cross-platform file locking — there
    is no portable advisory lock on Windows, and a lock that silently does
    nothing is worse than none.
    """

    def __init__(self, root: Path | str, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.dir = self.root / run_id
        self.rows_dir = self.dir / ROWS_DIR
        self._fh: Any = None
        self._part: Path | None = None
        self._written = 0
        self._since_sync = 0
        self._stats: Counter[str] = Counter()
        self._arm_params: dict[str, set[str]] = {}

    # -- state ---------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST_NAME

    @property
    def is_sealed(self) -> bool:
        return self.manifest_path.exists()

    @property
    def exists(self) -> bool:
        return self.dir.exists()

    def part_files(self) -> list[Path]:
        if not self.rows_dir.exists():
            return []
        return sorted(self.rows_dir.glob("part-*.jsonl"))

    # -- writing -------------------------------------------------------------

    def open(self, config: dict[str, Any] | None = None) -> "RolloutStore":
        """Create the run directory and open this process's part file.

        Writing to an already-sealed run is refused. A sealed run has a
        manifest whose checksums and counts describe the rows as they were;
        appending after that silently invalidates every number computed from
        the manifest.
        """
        if self.is_sealed:
            raise StoreError(
                f"run {self.run_id} is sealed and cannot be appended to; "
                f"a re-run is a new run_id, never an overwrite"
            )
        self.rows_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.dir / CONFIG_NAME
        if config is not None and not config_path.exists():
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )

        # pid plus a random suffix: pids are reused, and two sweeps started in
        # the same second on a recycled pid would otherwise share a part file
        # and interleave their writes.
        shard = f"{os.getpid():06d}-{int(time.time()):010d}-{os.urandom(3).hex()}"
        self._part = self.rows_dir / f"part-{shard}.jsonl"
        # `newline=""` keeps Python from translating the "\n" we write into
        # "\r\n" on Windows, so a part file's sha256 is the same on every OS.
        self._fh = self._part.open("a", encoding="utf-8", newline="")
        return self

    def append(self, gen: Generation) -> None:
        """Append one generation. Never rewrites, never reorders."""
        if self._fh is None:
            raise StoreError("store is not open; call open() before append()")
        row = gen.to_row()
        self._fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        self._written += 1
        self._since_sync += 1

        self._stats[f"arm:{gen.arm}"] += 1
        self._stats[f"finish:{gen.finish_reason}"] += 1
        self._stats[f"extract:{gen.extract_strategy}"] += 1
        if gen.truncated:
            self._stats["truncated"] += 1
        if not gen.tokens_exact:
            self._stats["tokens_approx"] += 1
        self._arm_params.setdefault(gen.arm, set()).add(gen.params_hash)

        if self._since_sync >= _FSYNC_EVERY:
            self.flush()

    def append_many(self, gens: Iterable[Generation]) -> int:
        n = 0
        for gen in gens:
            self.append(gen)
            n += 1
        return n

    def flush(self) -> None:
        """Push buffered rows to disk so a crash loses nothing already written."""
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

    # -- sealing -------------------------------------------------------------

    def seal(self, *, config: dict[str, Any] | None = None,
             extra: dict[str, Any] | None = None,
             write_parquet: bool = True) -> dict[str, Any]:
        """Write `_MANIFEST.json`, making the run valid to read.

        Counts are recomputed from the files on disk rather than taken from
        this process's counters, because a resumed run has rows written by an
        earlier process that this one never saw. A manifest describing only
        the current process's contribution would understate the run.
        """
        self.close()
        if self.is_sealed:
            raise StoreError(f"run {self.run_id} is already sealed")

        parts = self.part_files()
        files: list[dict[str, Any]] = []
        stats: Counter[str] = Counter()
        arm_params: dict[str, set[str]] = {}
        total = 0

        for part in parts:
            rows = 0
            for row in _read_jsonl(part):
                rows += 1
                stats[f"arm:{row.get('arm')}"] += 1
                stats[f"finish:{row.get('finish_reason')}"] += 1
                stats[f"extract:{row.get('extract_strategy')}"] += 1
                if row.get("finish_reason") == "length":
                    stats["truncated"] += 1
                if not row.get("tokens_exact", True):
                    stats["tokens_approx"] += 1
                arm_params.setdefault(str(row.get("arm")), set()).add(
                    str(row.get("params_hash"))
                )
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

        # Dirty at seal time disqualifies the run even when the id looks clean.
        # The id is stamped when the sweep opens; if the worktree was edited
        # while it ran, the recorded sha no longer describes the code that
        # produced the later rows.
        dirty_now = is_dirty()

        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "schema_version": SCHEMA_VERSION,
            "sealed_at": _utc_now(),
            "publishable": is_publishable(self.run_id) and not dirty_now,
            "n_rows": total,
            "files": files,
            "parquet": PARQUET_NAME if parquet_written else None,
            "counts": {
                "by_arm": {k[4:]: v for k, v in sorted(stats.items())
                           if k.startswith("arm:")},
                "by_finish_reason": {k[7:]: v for k, v in sorted(stats.items())
                                     if k.startswith("finish:")},
                "by_extract_strategy": {k[8:]: v for k, v in sorted(stats.items())
                                        if k.startswith("extract:")},
            },
            # The two rates worth watching between runs. A truncation rate that
            # moves is a silent accuracy regression: `finish_reason == "length"`
            # grades as a failure and looks exactly like a capability gap.
            "truncation_rate": (stats["truncated"] / total) if total else 0.0,
            "approx_token_rate": (stats["tokens_approx"] / total) if total else 0.0,
            # One params_hash per arm, always. More than one means rows from two
            # different configurations share a run_id, which must never happen.
            "params_hash_by_arm": {k: sorted(v) for k, v in sorted(arm_params.items())},
            "environment": {
                "git_sha7": git_sha7(),
                "dirty": dirty_now,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }
        if config is not None:
            manifest["config"] = config
        if extra:
            manifest["extra"] = extra

        # Written last, and only once every other artifact is on disk. The
        # manifest's existence is the seal, so it must never appear before the
        # thing it describes is complete.
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8", newline="",
        )
        tmp.replace(self.manifest_path)
        return manifest

    def _write_parquet(self, parts: list[Path]) -> bool:
        """Derive the Parquet view R3 and R4 read. Optional by design."""
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError:
            return False

        rows = [row for part in parts for row in _read_jsonl(part)]
        if not rows:
            return False
        # `extra` holds backend-specific keys whose types vary between rows;
        # Arrow would infer a struct from the first row and then fail on the
        # rest. Serializing it keeps the column stable and the data present.
        for row in rows:
            if isinstance(row.get("extra"), dict):
                row["extra"] = json.dumps(row["extra"], default=str, sort_keys=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, self.rows_dir / PARQUET_NAME, compression="zstd")
        return True

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "RolloutStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a part file, skipping a torn final line.

    A sweep killed mid-write leaves a partial last line. Skipping it is
    correct: resume will regenerate that cell, because an unparseable line
    contributes nothing to the completed set.
    """
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
    """Load a run's manifest, or fail because the run is not valid yet."""
    path = Path(root) / run_id / MANIFEST_NAME
    if not path.exists():
        raise UnsealedRunError(
            f"run {run_id} has no {MANIFEST_NAME} and is not valid to read; "
            f"it is either still running or was interrupted"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(root: Path | str, run_id: str, *,
              allow_unsealed: bool = False) -> Iterator[dict[str, Any]]:
    """Stream a run's rows.

    Refuses unsealed runs unless asked. `allow_unsealed=True` exists for
    exactly one caller — the resume index, which must read a run precisely
    because it is incomplete.
    """
    store = RolloutStore(root, run_id)
    if not store.exists:
        raise StoreError(f"no such run: {store.dir}")
    if not store.is_sealed and not allow_unsealed:
        raise UnsealedRunError(
            f"run {run_id} is not sealed; readers skip unsealed runs. Pass "
            f"allow_unsealed=True only if you are resuming it."
        )
    for part in store.part_files():
        yield from _read_jsonl(part)


def read_generations(root: Path | str, run_id: str, *,
                     allow_unsealed: bool = False) -> Iterator[Generation]:
    """Stream a run's rows as `Generation` objects."""
    for row in read_rows(root, run_id, allow_unsealed=allow_unsealed):
        yield Generation.from_row(row)


def list_runs(root: Path | str, *, sealed_only: bool = True) -> list[str]:
    """Run ids under `root`, newest-looking last.

    Sealed-only by default, matching the rule that readers skip unsealed runs.
    Nothing here resolves a "latest" — this lists what exists so a human can
    pin one explicitly.
    """
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
