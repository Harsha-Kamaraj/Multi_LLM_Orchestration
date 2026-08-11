# Cost-Aware LLM Orchestration

A learned policy decides **which model to spend on** — answer directly with a small
model, escalate to a large one, retry with repair, or decompose — optimizing a
tradeoff between accuracy, compute cost, and latency.

Correctness is decided by **executing unit tests in a sandbox**, never by asking
another model. Every reported number carries a confidence interval and is
reproducible from a `run_id`.

---

## The claim we're testing

Routing between a cheap and an expensive model beats always using either one, at
matched cost.

The honest version of that claim has to beat one specific baseline:
**`verifier_gated_cascade`** — run the small model, execute the visible tests,
escalate only on failure. It is strong, because *observing* failure beats
*predicting* it. Any framing that omits it is dishonest, so it's in the baseline
set from day one.

The central empirical question is where in the pipeline the signal actually lives:

| Decision point | Information available | Expected AUC |
|---|---|---|
| **D0** — before generating | Prompt only | 0.60 – 0.68 |
| **D1** — after generating | Prompt + candidate code + self-consistency + visible-test outcome | 0.80 – 0.90 |

That asymmetry is the most important number in the project, and it drives the
architecture. If `AUC_D1 < 0.75`, the premise is false and we stop — that gate is
in [ROADMAP.md](./ROADMAP.md).

---

## Start here

| If you are… | Read |
|---|---|
| New to the repo | This file, then [ROLES.md](./ROLES.md) |
| About to commit | [CONTRIBUTING.md](./CONTRIBUTING.md) — **enforced by a hook, not by trust** |
| Planning the week | [ROADMAP.md](./ROADMAP.md) |
| Looking for your job | Your own doc, below |

### Role docs

Each seat has a working doc: mandate, the interface it must satisfy, a week-1
checklist, a definition of done, and the failure modes specific to that seat.

| Role | Owner | Doc | Owns |
|---|---|---|---|
| R1 · Serving & Workers | Guru | [guru.md](./guru.md) | The GPUs |
| R2 · Verifier & Data | Diya | [diya.md](./diya.md) | Correctness |
| R3 · Policy & Learning | Harsha | [harsha.md](./harsha.md) | The policy |
| R4 · Evaluation & Analysis | Vivian | [vivian.md](./vivian.md) | The verdict |

Each doc ends with a **"what you must not do"** list. Boundaries are stated from
both sides on purpose — R1's says *don't grade*, R2's says *don't parse raw output*.

---

## How it fits together

```
tasks + splits          (R2 corpus, R4 splits)
        │
        ▼
   sweep on vLLM  ─────────────────────────────►  Generation
        │  (R1)                                       │
        │                                             ▼
        │                                    sandboxed grading  (R2)
        │                                             │
        └──────────────────►  rollout store  ◄────────┘
                              (Parquet, append-only,
                               partitioned by run_id)
                                     │
                       ┌─────────────┴─────────────┐
                       ▼                           ▼
              value heads + λ sweep  ──►  baselines + statistics
                     (R3)                        (R4)
                       │                           │
                 decisions.parquet            results.json
```

Two properties make this work as a four-person project:

**Only R1 needs GPUs. Only R2 needs sandboxing.** R3 and R4 need neither, so they
are never blocked on hardware.

**R3 and R4 develop against a synthetic rollout generator from day 1.** It emits
schema-valid rows with a *planted signal of known strength* and a known-optimal
policy — so the policy code and the statistics code can both be tested against an
answer that is correct by construction. The real store lands in week 2 and the
switch is a path change. If it needs a code change, the contract was violated.

---

## Ground rules

**Runs are immutable.** `run_id = {date}-{git_sha7}-{config_hash6}`. A run is
invalid until `_MANIFEST.json` is written; readers skip unsealed runs. A re-grade
is a new `run_id`, never an overwrite. Dirty worktrees stamp `-dirty` and are
non-publishable.

**Nothing reads "latest".** Every consumer pins an explicit `run_id`.

**Hidden tests are labels, never features** — including transitively, via anything
derived from pass rates.

**The test split is opened exactly once**, by R4, after the analysis is
pre-registered.

**No bare means.** Every number in the report carries an interval. This is a merge
blocker.

---

## Setup

```sh
git clone https://github.com/Harsha-Kamaraj/Multi_LLM_Orchestration
cd Multi_LLM_Orchestration

git config core.hooksPath .githooks    # required — enforces commit rules
pip install -e .
```

The hook checks commit format (`type: message`, 4–6 words), rejects co-authoring
trailers, and blocks commits that span more than one role's paths. Set it up before
your first commit, not after it rejects you.

> **Docker is required** for any graded number. The subprocess grader backend exists
> for local iteration only and **must never produce reported numbers**.

---

## Stack

Qwen2.5-Coder on **vLLM** · Docker sandbox · scikit-learn + LightGBM (CPU) ·
Parquet + DuckDB · SciPy / statsmodels · Hydra + Justfile

Deliberately **not** used: LangChain, LangGraph, CrewAI, AutoGen, a vector DB, or
an RL framework. The policy chooses among a handful of arms using a few dozen
features over ~1,000 tasks. A tabular model is the right tool until a measured gap
says otherwise.
