"""Frozen train/val/test splits — assigned by hash, not by shuffle.

R4 owns the split because it is an evaluation artifact: the person who proves
the result controls what is frozen. This module is how that ownership becomes
mechanical instead of a claim.

## Why hashing, not shuffling

The obvious implementation is `rng.permutation(tasks)` sliced 60/20/20. It is
wrong here, and the failure is silent.

The corpus grows. The pilot is 200 tasks; Phase 1 needs ~1000. Under a shuffle,
adding 800 tasks reassigns *every existing task* — a task that was in `train`
during the pilot lands in `test` for the real run, after models were tuned
while it was visible. The split file still looks fine. The contamination is
invisible and unrecoverable after the fact.

Hashing each `task_id` independently makes assignment a property of the task
rather than of the corpus it arrived in. Add 800 tasks and the original 200
keep their splits exactly. Remove some, reorder the file, regenerate the
manifest on another machine — same answer.

## Why the manifest is verifiable rather than merely stored

Every assignment is recomputable from `(salt, task_id)`, so `verify` can check
that the committed file is what the rule produces. A hand-edited split — moving
one stubborn task out of `test` — is caught. A stored-only mapping could not
detect that, and it is exactly the edit someone makes at 2am in week 9.

The corpus content hash is recorded too: if R2 regenerates `data/tasks/` in
place, the tasks behind these ids are no longer the tasks that were split, and
that is a different experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SPLITS: tuple[str, ...] = ("train", "val", "test")
DEFAULT_RATIOS: tuple[float, float, float] = (0.6, 0.2, 0.2)

# Resolution of the hash bucket. 10^6 makes the quantization error on a ratio
# negligible next to the sampling noise of any corpus we will ever build.
_BUCKETS = 1_000_000


class SplitError(ValueError):
    """The split manifest and the corpus disagree."""


def assign(task_id: str, *, salt: str, ratios: Sequence[float] = DEFAULT_RATIOS) -> str:
    """Which split a task belongs to. A pure function of the id and the salt.

    Independent of corpus size, corpus order, and every other task — which is
    the whole point. Two people on two machines with two different corpora get
    the same answer for a shared id.
    """
    digest = hashlib.sha256(f"{salt}\x00{task_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % _BUCKETS
    position = bucket / _BUCKETS

    cumulative = 0.0
    for name, ratio in zip(SPLITS, ratios):
        cumulative += ratio
        if position < cumulative:
            return name
    return SPLITS[-1]  # floating-point slack at the top edge


def corpus_hash(tasks: Iterable[Mapping[str, Any]]) -> str:
    """Content hash over the corpus, order-independent.

    Covers the fields a split actually depends on. If R2 regenerates
    `data/tasks/` in place and a prompt changes, the tasks behind these ids are
    not the tasks that were split — a different experiment wearing the same
    manifest.
    """
    parts = sorted(
        hashlib.sha256(
            "\x00".join([
                str(t.get("task_id", "")),
                str(t.get("prompt", "")),
                str(t.get("visible_tests", "")),
                str(t.get("hidden_tests", "")),
            ]).encode("utf-8")
        ).hexdigest()
        for t in tasks
    )
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    """A frozen split. Committed, hashed, and verifiable."""

    name: str
    salt: str
    ratios: tuple[float, float, float]
    corpus_hash: str
    counts: dict[str, int]
    n_tasks: int
    task_ids: dict[str, str] = field(default_factory=dict)

    def split_of(self, task_id: str) -> str:
        """Assignment for a task, recomputed rather than looked up.

        Recomputation is deliberate: it means a task absent from the manifest
        still gets the right answer when the corpus grows, and it means the
        stored map is a cache that `verify` can check rather than a source of
        truth nobody can audit.
        """
        return assign(task_id, salt=self.salt, ratios=self.ratios)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "salt": self.salt,
            "ratios": list(self.ratios),
            "corpus_hash": self.corpus_hash,
            "counts": dict(sorted(self.counts.items())),
            "n_tasks": self.n_tasks,
            "task_ids": dict(sorted(self.task_ids.items())),
        }

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.as_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return out

    @staticmethod
    def read(path: str | Path) -> "SplitManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return SplitManifest(
            name=data["name"],
            salt=data["salt"],
            ratios=tuple(data["ratios"]),  # type: ignore[arg-type]
            corpus_hash=data["corpus_hash"],
            counts=dict(data["counts"]),
            n_tasks=int(data["n_tasks"]),
            task_ids=dict(data.get("task_ids", {})),
        )


def build(
    tasks: Sequence[Mapping[str, Any]],
    *,
    name: str,
    salt: str,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> SplitManifest:
    """Assign every task and record the result.

    `salt` is the only knob, and changing it reshuffles everything — which is
    why it is committed rather than passed at call time. Re-salting after
    seeing results is the split-level equivalent of unblinding twice.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise SplitError(f"ratios must sum to 1.0, got {sum(ratios)}")
    if len(ratios) != len(SPLITS):
        raise SplitError(f"expected {len(SPLITS)} ratios, got {len(ratios)}")
    if not salt.strip():
        raise SplitError("salt must be non-empty; it is the split's identity")

    ids = [str(t["task_id"]) for t in tasks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise SplitError(
            f"corpus contains duplicate task_ids (e.g. {sorted(duplicates)[0]}); "
            f"a duplicated task would appear in its split twice and narrow "
            f"every interval"
        )

    mapping = {i: assign(i, salt=salt, ratios=ratios) for i in ids}
    counts = {s: sum(1 for v in mapping.values() if v == s) for s in SPLITS}
    return SplitManifest(
        name=name,
        salt=salt,
        ratios=tuple(ratios),  # type: ignore[arg-type]
        corpus_hash=corpus_hash(tasks),
        counts=counts,
        n_tasks=len(ids),
        task_ids=mapping,
    )


def ratio_tolerance(ratio: float, n: int, *, sigmas: float = 3.5) -> float:
    """How far a hash-assigned proportion may drift before it is suspicious.

    A fixed tolerance is the wrong tool. Assignment is an unbiased Bernoulli
    draw per task, so the standard error is `sqrt(p(1-p)/n)` and the acceptable
    drift *shrinks as the corpus grows*. A constant 6% would wave through a
    genuinely broken split at n=4000 while failing a perfectly healthy one at
    n=200 — which is exactly what it did the first time this was written.
    """
    if n <= 0:
        return 1.0
    standard_error = (ratio * (1.0 - ratio) / n) ** 0.5
    # A small floor so a tiny corpus does not demand impossible precision, and
    # a rounding allowance for the bucket quantization.
    return max(sigmas * standard_error, 0.01)


def verify(
    manifest: SplitManifest,
    tasks: Sequence[Mapping[str, Any]],
    *,
    require_same_corpus: bool = True,
    sigmas: float = 3.5,
) -> list[str]:
    """Check a committed manifest against a corpus. Empty list means clean.

    Returns problems rather than raising so a caller can report all of them —
    a drifted split usually trips several, and the combination identifies the
    cause faster than the first one alone.
    """
    problems: list[str] = []
    ids = [str(t["task_id"]) for t in tasks]

    if require_same_corpus:
        actual = corpus_hash(tasks)
        if actual != manifest.corpus_hash:
            problems.append(
                f"corpus content hash changed: manifest {manifest.corpus_hash[:12]} "
                f"vs corpus {actual[:12]}. The tasks behind these ids are not the "
                f"tasks that were split — that is a different experiment."
            )

    # The check a stored-only mapping cannot make: is the committed file what
    # the rule produces? This is what catches a hand-edited split.
    tampered = [
        task_id
        for task_id, stored in manifest.task_ids.items()
        if stored != assign(task_id, salt=manifest.salt, ratios=manifest.ratios)
    ]
    if tampered:
        problems.append(
            f"{len(tampered)} stored assignments do not match the salted hash "
            f"(e.g. {tampered[0]}). The manifest was edited by hand."
        )

    missing = [i for i in ids if i not in manifest.task_ids]
    if missing:
        problems.append(
            f"{len(missing)} corpus tasks are absent from the manifest "
            f"(e.g. {missing[0]}). Rebuild it — assignments for existing tasks "
            f"will not move."
        )

    for split, ratio in zip(SPLITS, manifest.ratios):
        observed = manifest.counts.get(split, 0) / max(manifest.n_tasks, 1)
        allowed = ratio_tolerance(ratio, manifest.n_tasks, sigmas=sigmas)
        if abs(observed - ratio) > allowed:
            problems.append(
                f"{split} holds {observed:.1%} of tasks, target {ratio:.0%}, "
                f"beyond {sigmas:g} standard errors ({allowed:.1%}) at "
                f"n={manifest.n_tasks}. Hash assignment is unbiased, so a drift "
                f"this large means the ratios or the salt changed."
            )
    return problems


def apply_to_tasks(
    manifest: SplitManifest, tasks: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return the corpus with a `split` field attached to every task."""
    return [{**t, "split": manifest.split_of(str(t["task_id"]))} for t in tasks]


def read_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Read R2's JSONL task manifest."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    if not out:
        raise SplitError(f"no tasks in {path}")
    return out
