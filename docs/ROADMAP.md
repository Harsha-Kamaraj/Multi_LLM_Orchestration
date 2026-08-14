# Roadmap

Ten weeks, four people, four phases with explicit gates. A phase does not start
until the previous gate passes; a failed gate changes the plan rather than being
retried indefinitely.

Full rationale lives in the design doc, which is distributed separately and is not
tracked in this repo. Per-role breakdowns: [guru](./guru.md) · [diya](./diya.md) ·
[harsha](./harsha.md) · [vivian](./vivian.md).

---

## Phase 0 — Contracts & feasibility · Week 1

Establish the shared schemas and find out whether the premise holds before
anyone commits four weeks to it.

**Deliverables — status at 13 Aug 2026**

| | Deliverable | Owner |
|---|---|---|
| ✅ | `schemas/` frozen: `Task` and `Rollout` JSON schemas | R4 |
| ✅ | Synthetic rollout generator emitting schema-valid rows with a planted, known-strength signal | R4 |
| ❌ | Sign-off from all four roles on the frozen schema — landed unilaterally, past the day-3 deadline | all |
| ✅ | 200-task pilot sweep (small ×3 seeds, large ×3 seeds), fully graded — `2026-08-13-c76a55d-4f4767`, 1200 rows generated and graded in Docker, 826 solved, `A_large − A_small = 19.33pp` against an ≥8pp gate | R1 · R2 |
| ❌ | Oracle-gap study — `eval.oracle_headroom()` is implemented, but there is no pilot to run it on | R4 |
| ❌ | Power calculation off measured discordance rates — `eval.mcnemar_sample_size()` exists; the discordance rate does not | R4 |

**Gate — measure all four, hard stop if `AUC_D1` fails**

| Measured | Quantity | Definition | Threshold | If it fails |
|---|---|---|---|---|
| ✅ | `A_large − A_small` | pass@1 difference between arms | ≥ 8 pp | **Measured 19.33 pp** (59.17% vs 78.50%, n=600 each). Arms are differentiated; no need to shrink the small model |
| ❌ | `A_oracle − A_large` | headroom above the best single arm | ≥ 5 pp | Large model dominates; only cost savings available, not accuracy |
| ❌ | `AUC_D0` | predicting "small solves" from **prompt-only** features | ≥ 0.65 | Pre-generation routing is dead. Move to D1 — expected, not fatal |
| ❌ | `AUC_D1` | same, from **post-generation** features | ≥ 0.75 | **Hard stop.** Neither decision point has signal; the premise is false |

Expect `AUC_D0 ≈ 0.60–0.68` and `AUC_D1 ≈ 0.80–0.90`. That asymmetry is the most
important number in the project and it should drive the architecture.

**Phase 0 has not passed.** One of four gate quantities now has a value —
`A_large − A_small` = 19.33 pp, clearing its 8 pp threshold on the graded pilot.
The other three do not: the oracle gap needs R4's headroom study, and both AUC
figures need R3's feature builders and value heads. `AUC_D1` is the hard stop,
and it remains unmeasured, so nothing below is formally unblocked.

---

## Phase 1 — MVP frontier · Weeks 2–4

The minimum system that produces a defensible result. Nothing outside this list ships.

**Deliverables — status at 13 Aug 2026**

| | Deliverable | Owner |
|---|---|---|
| ✅ | Qwen2.5-Coder-1.5B and 7B, **TP=1 both**, bf16 — both backends driven against real vLLM 0.11.0. One card on this machine, so the arms ran sequentially rather than one per GPU | R1 |
| ✅ | Characterization pass producing cost coefficients, `warm_latency_s`, `cold_start_s` — `bench/cost_coefficients.json` committed; `cold_start_s` 28.20s small / 55.36s large | R1 |
| ❌ | 1,000 tasks from MBPP+ / HumanEval+, split 60/20/20, manifest hashed and committed — the 200-task pilot corpus ✅ exists and is hashed; the full 1,000 and the `data/splits/` manifest ❌ do not | R2 · R4 |
| ✅ | Frozen-ladder sweep: 6 generations per task, all graded, all logged — 1200 rows = 200 × 2 arms × 3 seeds, sealed, graded, and written to Parquet alongside JSONL | R1 · R2 |
| ❌ | Calibrated `P(pass \| x, arm)` at D0 and D1, plus cost and latency regressors — `src/orchestrator/policy/` ✅ exists with a row contract and a label-stripping reader; the feature builders and value heads ❌ are not written | R3 |
| ❌ | λ-sweep producing the cost–accuracy frontier — the evaluator side (`eval.sweep`, `eval.pareto_front`) is ✅; the policy that feeds it is ❌ | R3 |
| ✅ | Seven baselines: `always_small`, `always_large`, `random_route(p)`, **`heuristic_route`**, `best_of_n_small`, `verifier_gated_cascade`, `oracle_router` — all implemented against fixtures | R4 |
| ✅ | Paired bootstrap CIs and McNemar tests — implemented and validated on the planted signal | R4 |
| ❌ | …run on the frozen test split, opened exactly once — `unlock_test_split()` guards it; the split does not exist and has never been opened | R4 |

**Gate:** policy beats `verifier_gated_cascade` on at least one frontier region,
with a confidence interval excluding zero. **Not evaluated** — there is no policy
and no rollout store.

**Second gate, equally binding:** `learned_D0` beats a *properly tuned*
`heuristic_route` on the same region. If it doesn't, the learned policy hasn't
earned its complexity at D0 — report that, and let D1 carry the result.

---

## Phase 2 — Repair & probe · Weeks 5–7

**Status: ❌ not started.** R1 has the `probe_small` and `repair_small` /
`repair_large` arms registered early, so the arm registry will not be reshaped
mid-project — but nothing has been run through them.

**Deliverables**

- Repair ladder (one round, seeded from each small-model sample)
- Self-consistency probe arm — spend *k*=3 cheap samples to buy a difficulty signal
- Calibration report (ECE, reliability diagrams) for every value head
- Full failure taxonomy, auto-tagged, validated against 100 hand-labelled samples
  with inter-rater agreement measured on 25

**Gate:** repair pays for itself — Δaccuracy / Δcost beats escalation in some λ region.

---

## Phase 3 — Hardening & replication · Weeks 8–10

**Status: ❌ not started**, with one exception — the reporting half of "Final
report" exists ahead of schedule: `eval.build()` emits a deterministic
`results.json` and `orch-eval report` formats it. It has no numbers to report.

**Deliverables**

- LiveCodeBench post-cutoff replication (contamination-controlled split)
- Ablations: feature groups, calibration on/off, ladder depth, heuristic families
- Session-based demo CLI — load, serve a handful of requests, unload
- Final report

**Gate:** the result replicates on the contamination-controlled split. If the routing
win exists only on the contaminated split, that is the actual finding — report it.

---

## Explicit non-goals

Cut deliberately, and each is defensible in one sentence:

| Cut | Why |
|---|---|
| `decompose` arm | Needs a multi-step corpus we don't have, and it is the fragile multi-agent pipeline this project exists to avoid |
| Online router + shadow A/B | Requires two models resident and warm; violates the on-demand constraint |
| LinUCB / online exploration | Same residency problem, plus it needs live traffic we don't have |
| Fine-tuned policy LM, GRPO, full MDP | Using billions of parameters to choose among three arms, before the tabular policy has been shown to fail |

The honest limitation to state up front: **every result is offline replay.** That is
how routing research is normally done, and saying it first is better than being asked.

---

## Critical path and parallelism

- **Week 1 is a hard serialization point.** Schemas must be frozen by day 3 — every
  downstream role builds against them. A schema change in week 5 costs the team a week.
- **R2's task manifest blocked R1's sweep** and nothing else. It was the only
  true cross-role dependency in Phase 1, it **cleared on 13 Aug**, and the sweep
  ran the same day: `2026-08-13-c76a55d-4f4767`, 1200 rows, sealed. The
  dependency is fully discharged in both directions: R2 graded the pilot the
  same night, so the **pilot now runs end to end** — generated, graded, costed,
  and schema-validated. That is the pipeline closing, **not the Phase 0 gate
  passing**: one of its four quantities is measured and three are not. No
  remaining blocker is a handoff or a GPU; the rest is R3's and R4's code.
- **R3 and R4 never block on GPUs.** They develop against the synthetic rollout
  generator, which doubles as a correctness test for the policy and stats code —
  the right answer is known by construction.
- **The real rollout store lands in week 2.** R3/R4 switch to it by changing one path.
  If that switch requires code changes, the contract was violated.

```
week 1     ALL ──── schema freeze (day 3)
                     │
           R2 ───── tasks + splits ────┐
                                       ▼
week 2     R1 ───── sweep ────── rollout store ─────┐
                                                    ▼
weeks 2-4  R3 ── value models ── decisions ──► R4 ── report
           (fixtures until week 2, then real data, no code change)
```

---

## Slippage policy

If Phase 1 slips past week 5, cut the probe arm and the online router — **not the
statistics**. A smaller result reported rigorously is worth more than a larger result
reported loosely, both for the product decision and for the portfolio.

If the schedule collapses entirely and only one thing survives, make it the
**evaluation harness with the six baselines and paired confidence intervals**.
