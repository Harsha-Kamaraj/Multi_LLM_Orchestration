# Team structure

Four roles, cut so each owns a distinct resource and a distinct failure mode.
Only R1 needs GPUs. Only R2 needs sandboxing. R3 and R4 need neither — which is
what makes the split genuinely parallel rather than nominally parallel.

Full rationale in the [design doc](./cost-aware-llm-orchestration-design.html).

---

## R1 — Serving & Workers · Guru

**Owns the GPUs.** `orchestrator/workers/`, `bench/`

Responsibilities
- vLLM deployment and tuning (offline batch for sweeps, OpenAI server for characterization)
- Arm implementations and prompt templates
- The rollout sweep runner — resumable, one command
- The cost/latency characterization pass

| | |
|---|---|
| **Input** | `tasks/*.jsonl` + `splits.json` (from R2), arm specs |
| **Output** | `Generation` records + `cost_coefficients.json` |

```python
generate(task_id, arm, seed) -> Generation {
    text, model_id, params_hash,
    prefill_tokens, decode_tokens,
    wall_ms,              # recorded, but NOT the reported latency metric
    finish_reason
}
```

**Done when:** a sweep is one command, is resumable, and imputed latency correlates
with measured wall-clock at R² > 0.9.

> Sweeps and serving are different modes and must not be conflated. Sweeps use
> vLLM offline `LLM.generate` tuned purely for throughput; characterization and
> online serving use the OpenAI-compatible server at declared, fixed concurrency.
> Wall-clock from a batched sweep is **not** serving latency.

---

## R2 — Verifier & Data · Diya

**Owns correctness.** `orchestrator/graders/`, `data/`

Responsibilities
- The sandbox — Docker, `--network none --read-only --memory 512m --pids-limit 128 --cpus 1`
- The graders (pytest / EvalPlus semantics, per-test pass counts)
- Dataset ingestion, contamination filtering, split manifests
- The visible/hidden test split, enforced mechanically
- Reward-hack detection

| | |
|---|---|
| **Input** | Raw benchmarks; `(task, code)` pairs |
| **Output** | `Grade` records, `tasks/*.jsonl`, hashed `splits.json` |

```python
grade(task, code) -> Grade {
    visible_passed, visible_total,     # usable at inference by policy + cascade
    hidden_passed, hidden_total,       # labels only — never a feature
    error_class, hack_flags, duration_s
}
```

**Done when:** grading is a pure function, container-isolated, and a deliberately
malicious solution suite is caught by the hack detector.

> The visible/hidden split is what makes the D1 decision non-trivial. Enforce it in
> code, not by convention. A bare subprocess grader is fine for local iteration and
> must never produce reported numbers.

---

## R3 — Policy & Learning · Harsha

**No GPU.** `orchestrator/policy/`

Responsibilities
- Feature builders for D0 (pre-generation) and D1 (post-generation)
- Three value heads: `P(pass)`, `E[cost]`, `E[latency]`
- Calibration (isotonic), the λ-sweep, offline policy evaluation
- The decision rule

| | |
|---|---|
| **Input** | rollout store (Parquet) — nothing else |
| **Output** | `policy.pkl` + `decisions.parquet` per (policy, λ) |

```python
# λ is a runtime knob, not a training hyperparameter
def decide(x, lam):
    return argmax_a(P_pass[a](x) - lam * E_cost[a](x))
```

**Done when:** the policy recovers the planted signal in synthetic rollouts, and
`P(pass)` is calibrated to ECE < 0.05.

> Every feature must be computable at the decision point it claims. Hidden-test
> outcomes are labels, never features — including transitively, e.g. via a
> "difficulty" column that was itself derived from pass rates.

---

## R4 — Evaluation & Analysis · Vivian

**The referee.** `eval/`

Responsibilities
- All six baselines
- All statistics — cluster bootstrap, McNemar, BH correction, power analysis
- The failure taxonomy and the router confusion matrix
- The leakage audit
- The report

| | |
|---|---|
| **Input** | rollout store + `decisions.parquet` |
| **Output** | `results.json` + `report.html` |

**Done when:** every reported number carries a confidence interval, and re-running
the pipeline reproduces the report byte-identically from a `run_id`.

> R4 owns the frozen test split and is the **only** role permitted to open it.

---

## Why R3 and R4 are separate people

The engineer who trains the policy should not be the engineer who proves it won.
This removes the incentive to tune against the test split, and it means the leakage
audit is performed by someone with no stake in passing it. On a four-person team
this is the cheapest integrity control available — it costs nothing but the
discipline to not merge the roles when the schedule tightens.

---

## The shared contract

Two JSON schemas — `Task` and `Rollout` — plus a fixture generator, in a `schemas/`
package that every role imports and no role owns. Changes require sign-off from all
four. In exchange for that friction, each role develops and tests in complete
isolation from day 1.

```
rollout {
  // identity
  task_id, split, dataset, arm, seed, run_id, code_version,
  // generation (R1)
  text, model_id, params_hash, prefill_tokens, decode_tokens, wall_ms,
  // grading (R2)
  visible_passed, visible_total, hidden_passed, hidden_total,
  error_class, hack_flags, grade_duration_s,
  // imputed (R1 characterization)
  gpu_seconds, imputed_latency_s,
  // ladder position — makes sequential replay possible
  parent_rollout_id, ladder_step
}
```

Append-only, partitioned by `run_id`. Never mutated — a re-grade writes a new
`run_id`, so any published number can be reproduced from the exact rows that
produced it.

---

## Ownership at a glance

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| Needs GPUs | ✅ | | | |
| Needs sandboxing | | ✅ | | |
| Reads rollout store | | | ✅ | ✅ |
| Can open the test split | | | | ✅ |
| Blocked in week 1 by | R2's manifest | — | — | — |
