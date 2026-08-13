# Cost-Aware LLM Orchestration

A learned policy decides **which model to spend on** — answer directly with a small
model, escalate to a large one, or retry with repair — optimizing a tradeoff
between accuracy, compute cost, and latency.

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
in [ROADMAP.md](./docs/ROADMAP.md).

---

## The comparison

Seven baselines, arranged as a **capacity ladder**. Each rung varies exactly one
thing from the rung above it, so a gap between two rows attributes to a cause
rather than to "the whole system got better".

| Baseline | Free parameters | Information | Isolates |
|---|---|---|---|
| `always_small` | — | — | Cost floor |
| `always_large` | — | — | Single-arm accuracy ceiling |
| `random_route(p)` | 0 | none | Does routing at all help? |
| `heuristic_route` | 1–2 thresholds | prompt only | **Does learning beat human priors?** |
| `learned_D0` | dozens | prompt only | |
| `learned_D1` | dozens | prompt + code + visible tests | **Does observing beat predicting?** |
| `best_of_n_small` | — | — | Controls for "more samples helps", matched cost |
| `verifier_gated_cascade` | 0 | visible-test outcome | **The one to beat** |
| `oracle_router` | — | everything | Headroom available in principle |

Two adjacent comparisons carry most of the result:

- `heuristic_route` → `learned_D0` — **learning**, information held constant
- `learned_D0` → `learned_D1` — **information**, learning held constant

`heuristic_route` is tuned as hard as the policy is: thresholds fit on validation,
swept across λ to produce a frontier rather than a point. A baseline tuned less than
the method is a strawman, not a baseline. **If the tuned heuristic wins, that is the
finding** — the learned policy did not earn its complexity, and the report says so.

---

## Status — 13 Aug 2026

Phase 0, week 1. **Every number above is a target, not a measurement.** Nothing
has run on a GPU, no task corpus exists, and not one gate has been evaluated.

| Deliverable | Owner | State |
|---|---|---|
| `schemas/` — `Task` + `Rollout` contracts, version guards, validation | R4 | ✅ |
| Synthetic rollout generator with a planted signal + adversarial fixtures | R4 | ✅ |
| Sweep runner, rollout store, resume, cost model, code extraction | R1 | ✅ built · ❌ never on a GPU |
| Baselines, cluster bootstrap, McNemar, BH, matched-cost frontier, confusion matrix | R4 | ✅ |
| Leakage audit, results report, golden-run comparison, `orch-eval` CLI | R4 | ✅ |
| Task corpus `data/tasks/` — **blocks R1's sweep** | R2 | ❌ |
| Docker grader end to end, visible/hidden split, hack detection | R2 | ❌ |
| Frozen splits `data/splits/` | R4 | ❌ |
| Policy — feature builders, value heads, calibration, λ-sweep | R3 | ❌ |
| CI (`.github/workflows/`) | infra | ❌ |

**Phase 0 gate — all four quantities are unmeasured:**

| Quantity | Threshold | Measured |
|---|---|---|
| `A_large − A_small` | ≥ 8 pp | ❌ |
| `A_oracle − A_large` | ≥ 5 pp | ❌ |
| `AUC_D0` | ≥ 0.65 | ❌ |
| `AUC_D1` | ≥ 0.75 | ❌ |

479 tests pass — `bench/tests` (270), `eval/tests` (159), `schemas/tests` (50) —
with no GPU and no network, in ~60 s. That is a statement about code, not about
the claim this project exists to test.

---

## Start here

| If you are… | Read |
|---|---|
| New to the repo | This file, then [ROLES.md](./docs/ROLES.md) |
| About to commit | [CONTRIBUTING.md](./docs/CONTRIBUTING.md) — **enforced by a hook, not by trust** |
| Planning the week | [ROADMAP.md](./docs/ROADMAP.md) |
| Looking for your job | Your own doc, below |

### Role docs

Each seat has a working doc: mandate, the interface it must satisfy, a week-1
checklist, a definition of done, and the failure modes specific to that seat.

| Role | Owner | Doc | Owns |
|---|---|---|---|
| R1 · Serving & Workers | Guru | [guru.md](./docs/guru.md) | The GPUs |
| R2 · Verifier & Data | Diya | [diya.md](./docs/diya.md) | Correctness |
| R3 · Policy & Learning | Harsha | [harsha.md](./docs/harsha.md) | The policy |
| R4 · Evaluation & Analysis | Vivian | [vivian.md](./docs/vivian.md) | The verdict |

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

## Hardware and the shape it forces

**2× RTX 4090, 24 GB each. Nothing always-on, nothing distributed.**

These are consumer cards: **no NVLink, P2P disabled.** Tensor parallelism across
them over PCIe is slow and fragile, so the hard rule is **every model fits on one
card**. That single constraint picks the models.

| Model | bf16 weights | One 4090? |
|---|---|---|
| Qwen2.5-Coder-1.5B — **small arm** | ~3.1 GB | yes |
| Qwen2.5-Coder-7B — **large arm** | ~15.2 GB | yes, ~8 GB left for KV |
| Qwen2.5-Coder-14B | ~29.4 GB | no — would need AWQ |
| Qwen2.5-Coder-32B | ~65.6 GB | no — AWQ ~19 GB, tight |

**Both arms are bf16 on purpose.** Quantizing only the large arm would confound
model scale with quantization damage, and the measured accuracy gap would no longer
mean what the report claims it means. The 1.5B/7B gap is ~20pp on HumanEval+,
comfortably past the 8pp gate.

The second card buys **throughput, not size**: the small arm sweeps on GPU0 while
the large arm sweeps on GPU1 — two independent single-GPU vLLM processes, no
coordination, roughly half the wall-time.

### Why nothing needs to be always-on

The architecture is **log once, replay many times**. Rollouts are swept ahead of
time; every policy, baseline, and λ afterward is arithmetic over Parquet on CPU.
No two models are ever resident together, and no comparison requires a live server.

The hardware constraint and the design agree — that isn't a workaround, it's why
the frontier can be swept at decision time instead of retrained.

**Latency is therefore two numbers, never one:**

- `warm_latency_s` — steady state, model resident, batch=1. What the policy
  optimizes and what a deployment actually sees.
- `cold_start_s` — vLLM startup, 30–90 s. Measured once per model, reported
  separately, and **excluded from the routing objective** because any real
  deployment amortizes it.

One vLLM process per model **per sweep** — never per task.

---

## Stack

Qwen2.5-Coder (1.5B / 7B) on **vLLM** · Docker sandbox · scikit-learn + LightGBM
(CPU) · Parquet + DuckDB · SciPy / statsmodels · Hydra + Justfile

Deliberately **not** used: LangChain, LangGraph, CrewAI, AutoGen, a vector DB, or
an RL framework. The policy chooses among a handful of arms using a few dozen
features over ~1,000 tasks. A tabular model is the right tool until a measured gap
says otherwise.

**Explicit non-goals:** a `decompose` arm (needs a multi-step corpus we don't have,
and it's the fragile multi-agent pipeline this project exists to avoid), an online
router, and live A/B serving. Every result is offline replay — which is how routing
research is normally done, and is stated up front rather than discovered later.
