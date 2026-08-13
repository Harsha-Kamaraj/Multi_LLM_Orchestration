"""Loading R2's task manifest and R4's split manifest.

R1 reads both and owns neither. This module is deliberately strict about what
it accepts, because a malformed corpus discovered at hour nine of a sweep costs
GPU time, and every check here is cheap and runs before the first token.

The strictness that matters most is **duplicate `task_id`s**. A duplicate makes
the resume key ambiguous — two different tasks share a cell — so resume either
skips real work or overwrites it, and the store ends up with rows whose
`task_id` does not identify what was generated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..types import Task
from .errors import WorkerError

# Names R2 and R4 might reasonably use for the same thing. Accepted on read
# because the corpus is not R1's to define; normalized immediately so nothing
# downstream has to know which spelling arrived.
_SPLIT_KEYS = ("split", "split_name", "partition")
_TASK_ID_KEYS = ("task_id", "id", "name")
_PROMPT_KEYS = ("prompt", "text", "question", "description")
_TESTS_KEYS = ("tests", "test", "test_code", "test_list")
_ENTRYPOINT_KEYS = ("entrypoint", "entry_point", "func_name")


class CorpusError(WorkerError):
    """The task or split manifest is not usable as given."""


def _first(d: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _as_source(value: Any) -> str:
    """Normalize a test field that may arrive as a string or a list of lines."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read one JSON object per line, reporting the line that failed.

    A bare `JSONDecodeError` on a 1,000-line manifest is close to useless; the
    line number is what makes it a two-second fix.
    """
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(
                    f"{path}:{lineno}: not valid JSON ({exc.msg})"
                ) from exc


def load_tasks(path: Path | str, *, require_tests: bool = False) -> list[Task]:
    """Load tasks from a `.jsonl` file, or every `.jsonl` under a directory.

    `require_tests` is off by default because R1 does not need tests to
    generate — only R2 does. A sweep should not be blocked by a manifest whose
    hidden tests have not been attached yet.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise CorpusError(f"no .jsonl task files under {path}")
    elif path.exists():
        files = [path]
    else:
        raise CorpusError(
            f"no task manifest at {path}; R2 owns `data/tasks/` and the manifest "
            f"is R1's only cross-role dependency in Phase 1"
        )

    tasks: list[Task] = []
    seen: dict[str, str] = {}
    for file in files:
        for lineno, record in enumerate(iter_jsonl(file), start=1):
            task_id = _first(record, _TASK_ID_KEYS)
            if not task_id:
                raise CorpusError(
                    f"{file}:{lineno}: record has no task_id (looked for "
                    f"{list(_TASK_ID_KEYS)})"
                )
            task_id = str(task_id)
            if task_id in seen:
                raise CorpusError(
                    f"{file}:{lineno}: duplicate task_id {task_id!r}, already seen "
                    f"in {seen[task_id]}. Task ids key the resume index — a "
                    f"duplicate makes a completed cell ambiguous."
                )
            seen[task_id] = f"{file}:{lineno}"

            prompt = _first(record, _PROMPT_KEYS, "")
            if not str(prompt).strip():
                raise CorpusError(f"{file}:{lineno}: task {task_id!r} has no prompt")
            tests = _as_source(_first(record, _TESTS_KEYS))
            if require_tests and not tests.strip():
                raise CorpusError(f"{file}:{lineno}: task {task_id!r} has no tests")

            metadata = dict(record.get("metadata") or {})
            # Carry through the keys R1 actually reads, without copying the
            # whole record: anything else is R2's business and copying it
            # blindly is how a hidden-test field ends up somewhere it can be
            # rendered into a prompt.
            for key in ("visible_tests", "dataset", "source", "mock_solution"):
                if key in record and key not in metadata:
                    metadata[key] = record[key]
            split = _first(record, _SPLIT_KEYS)
            if split is not None:
                metadata["split"] = str(split)

            tasks.append(Task(
                task_id=task_id,
                prompt=str(prompt),
                tests=tests,
                entrypoint=str(_first(record, _ENTRYPOINT_KEYS, "") or ""),
                metadata=metadata,
            ))

    if not tasks:
        raise CorpusError(f"no tasks found in {path}")
    return tasks


def load_splits(path: Path | str) -> dict[str, str]:
    """Load `splits.json` as a `task_id -> split` mapping.

    Accepts either shape R4 might commit: a flat mapping, or a mapping from
    split name to a list of task ids.
    """
    path = Path(path)
    if not path.exists():
        raise CorpusError(f"no split manifest at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CorpusError(f"{path}: expected a JSON object")

    # `{"train": [...], "val": [...]}` — split to task ids.
    if data and all(isinstance(v, (list, tuple)) for v in data.values()):
        out: dict[str, str] = {}
        for split, ids in data.items():
            for task_id in ids:
                out[str(task_id)] = str(split)
        return out

    # `{"task_id": "train"}`, possibly nested under a container key. R4's
    # committed manifest uses `task_ids` and carries sibling metadata
    # (`corpus_hash`, `name`, `salt`) that are strings too — without checking
    # the container keys first, the flat branch below harvests that metadata
    # as if it were three task ids and every real task comes back unassigned.
    for container in ("splits", "task_ids"):
        if isinstance(data.get(container), dict):
            data = data[container]
            break
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


@dataclass(frozen=True)
class CorpusView:
    """Tasks plus the split each belongs to, ready for a sweep."""

    tasks: tuple[Task, ...]
    splits: dict[str, str]

    def split_of(self, task_id: str) -> str:
        return self.splits.get(task_id, "")

    @property
    def fingerprint(self) -> str:
        """Content hash of the corpus as this sweep will use it.

        Folded into the `run_id`, so regenerating the task manifest produces a
        new run rather than silently extending the old one. Hashing only the
        *path* would let R2 rewrite `data/tasks/pilot.jsonl` under a sweep and
        have the new tasks land in a directory whose manifest describes the
        previous corpus.

        Covers exactly the fields that reach a prompt or the extractor —
        prompt, entrypoint, visible tests — and not the hidden suite, which R1
        never renders and which R2 may legitimately attach later without
        changing what was generated.
        """
        h = hashlib.sha256()
        for task in sorted(self.tasks, key=lambda t: t.task_id):
            for part in (
                task.task_id,
                task.prompt,
                task.entrypoint,
                str(task.metadata.get("visible_tests", "")),
                self.split_of(task.task_id),
            ):
                h.update(part.encode("utf-8"))
                h.update(b"\x00")
        return h.hexdigest()[:12]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[self.split_of(task.task_id) or "unassigned"] = (
                counts.get(self.split_of(task.task_id) or "unassigned", 0) + 1
            )
        return dict(sorted(counts.items()))


def build_corpus(tasks_path: Path | str,
                 splits_path: Path | str | None = None,
                 *, include_splits: Iterable[str] | None = None,
                 limit: int | None = None) -> CorpusView:
    """Load tasks, attach splits, and optionally filter and truncate.

    `limit` takes the first N in manifest order rather than a random sample.
    A random pilot subset would differ between two runs of "the same" pilot,
    and the point of the pilot is that it can be repeated.
    """
    tasks = load_tasks(tasks_path)

    splits: dict[str, str] = {}
    manifest_splits: dict[str, str] = {}
    if splits_path is not None:
        manifest_splits = load_splits(splits_path)
        splits.update(manifest_splits)
    # A split carried on the task record itself is a fallback for the period
    # before R4's manifest exists. The manifest wins where both are present.
    for task in tasks:
        if task.task_id not in splits and task.metadata.get("split"):
            splits[task.task_id] = str(task.metadata["split"])

    # A split manifest that matches no task is a format mismatch, not an empty
    # split. Left silent it degrades into "everything unassigned", which makes
    # `--include-splits` inert and hides the fact that the test-split fence is
    # no longer fencing anything. Checked against the manifest's own mapping,
    # not the merged one: splits carried on the task records would otherwise
    # mask a manifest that contributed nothing.
    if splits_path is not None and not any(
        t.task_id in manifest_splits for t in tasks
    ):
        raise CorpusError(
            f"{splits_path}: split manifest assigns no task in {tasks_path}; "
            f"the two files disagree on task ids or the manifest shape is not "
            f"recognised"
        )

    if include_splits is not None:
        wanted = {str(s) for s in include_splits}
        tasks = [t for t in tasks if splits.get(t.task_id, "") in wanted]
        if not tasks:
            raise CorpusError(
                f"no tasks left after filtering to splits {sorted(wanted)}; "
                f"available splits: {sorted(set(splits.values()))}"
            )
    if limit is not None:
        tasks = tasks[:limit]

    return CorpusView(tasks=tuple(tasks), splits=splits)
