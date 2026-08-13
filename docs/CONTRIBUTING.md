# Contributing

Rules that aren't enforced aren't rules. Everything here is checked by
`.githooks/pre-commit`, CI, or `CODEOWNERS` — nothing relies on remembering.

Project overview in the [README](../README.md); your role's scope and week-1
checklist in your own doc — [guru](./guru.md) · [diya](./diya.md) ·
[harsha](./harsha.md) · [vivian](./vivian.md).

**Enable the hooks once, on clone:**

```sh
git config core.hooksPath .githooks
```

---

## Branches

No direct commits to `main`. Ever — including hotfixes.

*Bootstrap exception:* the initial scaffold commits predate the remote and any
branch protection. This rule takes effect the moment `main` is pushed — enable
branch protection in the same session, or the rule is decorative.

Branch names declare ownership so a stale branch is attributable at a glance:

```
r1/sweep-runner
r2/timeout-handling
r3/d1-features
r4/bootstrap-ci
```

PRs are **squash-merged**, so `main` is one commit per PR and every change is
revertible with a single `git revert`. Reviews are required; the reviewer is
whoever `CODEOWNERS` assigns.

---

## Commit format

```
type: message
```

**Message is 4–6 words.** Present tense, imperative, no trailing period.
Prefer many small commits over few large ones — if a message needs "and", or
runs past six words, it is two commits.

**No co-authoring trailers.** No `Co-Authored-By`, no `Signed-off-by`, no tool
attribution. Authorship is the committer.

**Types**

| Type | Use for |
|---|---|
| `feat` | New capability |
| `fix` | Corrects wrong behavior |
| `perf` | Same behavior, faster or cheaper |
| `refactor` | No behavior change |
| `test` | Tests only |
| `data` | Regenerating a dataset, manifest, or fixture |
| `exp` | An experiment run — **must include the `run_id` in the body** |
| `eval` | Evaluation logic, statistics, reporting |
| `docs` | Documentation |
| `chore` | Tooling, deps, CI |

**Scopes** are *not* written in the message. They exist only to enforce
isolation: the hook derives the scope of your staged paths and rejects a commit
that spans more than one.

| Scope | Owner | Paths |
|---|---|---|
| `workers` | R1 | `src/orchestrator/workers/`, `bench/` |
| `grader` | R2 | `src/orchestrator/graders/` |
| `data` | R2 | `data/tasks/` |
| `policy` | R3 | `src/orchestrator/policy/` |
| `eval` | R4 | `eval/` |
| `splits` | R4 | `data/splits/` |
| `schemas` | R4 (see below) | `schemas/` |
| `infra` | rotating | CI, tooling, root config |

Examples:

```
feat: add resumable sweep runner
fix: handle container timeout correctly
eval: add cluster bootstrap intervals
data: regenerate mbpp manifest
exp: sweep lambda across decades

    run_id: 2026-08-14-a3f91c2-7d4e08
```

---

## Every commit must be atomic, reversible, and scoped

**Atomic** — one logical change. If the message needs "and", split it.

**Reversible** — `git revert` must leave a working tree. Two consequences:

- Never mix a code change with the data it regenerates. Regeneration is its own
  `data(...)` or `exp(...)` commit carrying the `run_id`.
- Never mix a schema change with the code that consumes it. Schema lands first,
  additive; consumers follow.

**Scoped** — one scope per commit. The pre-commit hook derives scopes from your
staged paths and rejects multi-scope commits. This is what makes "no cross-role
edits" mechanically checkable rather than aspirational.

Allowed exception, because these genuinely co-change: `schemas` + `infra`
(a schema plus its fixture generator).

---

## Cross-role edits

You may not commit to another role's paths without their approval on the PR.
`CODEOWNERS` enforces this at review time; the pre-commit hook catches it at
commit time so you find out in two seconds rather than after pushing.

Need something from another role's surface? Open an issue against that scope.
Do not fork the behavior locally — a divergence discovered in week 5 costs more
than a two-day wait.

---

## Schema changes

Split by reversibility. This is the rule that keeps the contract meaningful
without deadlocking the team.

| Change | Approval | Why |
|---|---|---|
| Add an **optional** field | 1 reviewer | Additive, non-breaking, old readers unaffected |
| Rename, remove, retype, or change nullability | **All four roles** | Destructive; silently breaks readers |
| Bump `schema_version` | All four | Follows any destructive change |

Every rollout row carries `schema_version`. A mixed-version store must be
*detectable*, never silently averaged.

---

## Runs are immutable

A run directory is invalid until `_MANIFEST.json` is written. Readers skip
unsealed runs. Never overwrite a run — a re-grade is a new `run_id`.

Downstream consumers pin an explicit `run_id`. Nothing reads "latest".

`run_id` format: `{date}-{git_sha7}-{config_hash6}`. Runs from a dirty worktree
are stamped `-dirty` and are non-publishable.

---

## What blocks a merge

| | Check | Enforced by |
|---|---|---|
| ✅ | Pre-commit hook passes (scope, format, no cross-role edits) | `.githooks/pre-commit`, per clone |
| ✅ | `CODEOWNERS` review from every owning role | GitHub, if the listed handles have write access |
| ⚠️ | Golden-run test reproduces `results.json` exactly, or the change is marked `BREAKING-GOLDEN:` in the PR body | Machinery ✅ — `eval.build()` emits a deterministic report and `eval.compare_to_golden()` diffs it (`orch-eval golden`). Gate ❌ — **no golden file is committed**, so nothing is actually being compared. |
| ✅ | Adversarial-fixture pipeline test passes | `schemas/adversarial.py` + `schemas/tests/test_adversarial.py` |
| ⚠️ | Leakage canary caught | Machinery ✅ — `eval.audit()` runs seven checks including `check_canary`, independent of R3's code. Gate ❌ — R3 has produced no features to audit. |
| ❌ | No new bare mean in the report — every number carries an interval | There is no report yet. `eval/tests/test_integration.py` encodes the rule as a test on the harness, which is the closest thing that exists. |

### What is not enforced yet — 13 Aug 2026

Two rules in this file are currently decorative, and saying so is cheaper than
discovering it during a Phase 1 merge:

**❌ There is no CI.** No `.github/workflows/` directory exists. `CODEOWNERS`
already assigns all-four approval to that path, and this file says checks are
"checked by the hook, CI, or `CODEOWNERS`" — today it is the hook and
`CODEOWNERS` only. Every check marked ❌ above is a check nobody runs.

**❌ Branch protection is unverified.** The bootstrap exception below expired the
moment `main` was pushed. Until protection is on, "no direct commits to `main`"
is honour-system, and the hook does not enforce it — the hook checks scope and
format, never the branch.

The pre-commit hook is also opt-in per clone (`git config core.hooksPath
.githooks`). A contributor who skips that line is subject to none of this.
