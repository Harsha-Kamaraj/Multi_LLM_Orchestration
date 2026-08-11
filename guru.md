# R1 — Serving & Workers · Guru

You own the GPUs. Nobody else touches them, and nobody else is allowed to be
blocked waiting for them.

**Your paths:** `src/orchestrator/workers/`, `bench/`
**Your commit scope:** `workers`
**You block:** R2 (needs your code output to grade), R3/R4 (need the rollout store)
**You are blocked by:** R2's task manifest — and nothing else

---

## Mandate

Turn `(task_id, arm, seed)` into a `Generation` record, at throughput, reproducibly,
and produce the cost coefficients that let everyone else talk about money without
owning a GPU.

---

## The one distinction that defines this role

**Sweeps and serving are different modes. Never conflate them.**

| | Sweep | Characterization / serving |
|---|---|---|
| Interface | vLLM offline `LLM.generate` | vLLM OpenAI-compatible server |
| Tuned for | Throughput | Latency at fixed concurrency |
| Batch | As large as memory allows | Declared: batch=1 and batch=8 |
| Wall-clock is | **Meaningless as latency** | The measurement |

A batched sweep runs 256 requests concurrently. The wall-clock of any one of them
is a function of queue depth, not of the model. If you report sweep wall-clock as
latency, every latency number in the project is wrong and the error is invisible —
it looks like plausible data.

Log `wall_ms` anyway. It's a useful sanity signal. It is **not** the reported metric.

---

## Cost model — characterize once, impute everywhere

Three tiers, in order of authority:

1. **Tokens** — primary. Hardware-independent, reproducible on anyone's machine.
2. **GPU-seconds** — secondary. `prefill_tokens × a_model + decode_tokens × b_model`
3. **USD** — tertiary. GPU-seconds × an instance rate, stated as an assumption.

The coefficients `a` and `b` come from a **separate characterization pass** on the
OpenAI server at batch=1 and batch=8, never from the sweep. You fit them once per
model, commit them to `cost_coefficients.json`, and every rollout gets `gpu_seconds`
and `imputed_latency_s` computed from that table.

This is what makes the cost numbers survive a hardware change. Re-characterize on
new hardware; the rollouts don't need regenerating.

---

## Interface you must satisfy

```python
generate(task_id, arm, seed) -> Generation {
    text, model_id, params_hash,
    prefill_tokens, decode_tokens,
    wall_ms,              # recorded, NOT the reported latency metric
    finish_reason
}
```

`params_hash` covers temperature, top_p, max_tokens, and the prompt template. If a
prompt template changes, the hash changes, and old rollouts are distinguishable from
new ones. This is non-negotiable — without it, a mid-sweep prompt tweak silently
poisons the store and nobody finds out.

---

## Code extraction is yours, not R2's

You hand R2 **code**, never raw model output. R2 grades what it's given.

If extraction lived in the grader, changing a prompt template would break R2's tests
and you'd own a bug in someone else's file. Keeping the fence-parsing on your side
means prompt format is entirely your business.

---

## Week 1 (Phase 0)

- [ ] Sign off on `schemas/` by day 3 — this is a hard serialization point
- [ ] vLLM up, both models loading, `LLM.generate` producing schema-valid rows
- [ ] 200-task pilot sweep: small ×3 seeds, large ×3 seeds, handed to R2 for grading
- [ ] First characterization pass → `cost_coefficients.json`

Your Phase 0 number is `A_large − A_small ≥ 8pp`. If the arms aren't differentiated,
**shrink the small model** — that's the fix, and it's yours to make.

---

## The sweep runner

Three properties, all required:

**One command.** If running a sweep needs a checklist, it will be run inconsistently.

**Resumable.** Sweeps take hours and will be interrupted. Resume must be keyed on
`(task_id, arm, seed, params_hash)` — a completed cell is never recomputed, and a
cell computed under different params is never mistaken for a hit.

**Append-only output.** Partitioned by `run_id`. You never mutate a row. A re-run
under different conditions is a new `run_id`, full stop.

---

## Definition of done

A sweep is one command, is resumable, and **imputed latency correlates with measured
wall-clock at R² > 0.9**.

That R² is the honesty check on the whole cost model. If it's low, the imputation is
fiction and you must say so before R4 builds a frontier on top of it.

---

## Things that will burn you

**Silent truncation.** `finish_reason == "length"` is a failed generation dressed as
a successful one. It grades as a fail and looks like a model capability gap. Count
them, report the rate, and alarm if it moves.

**Non-determinism at temperature 0.** vLLM batching makes T=0 not bitwise
reproducible across batch sizes. Don't promise determinism you can't deliver — pin
seeds, log them, and treat the seed as part of the row's identity.

**Tokenizer drift.** `prefill_tokens` must come from the serving tokenizer, not an
estimate. Every cost number downstream is built on it.

---

## What you must not do

- Report sweep wall-clock as latency
- Change a prompt template without changing `params_hash`
- Mutate a row in the rollout store
- Grade anything — that's R2's, and the separation is what makes grading trustworthy
- Open the test split — R4 only

---

[README](./README.md) · [CONTRIBUTING](./CONTRIBUTING.md) · [ROADMAP](./ROADMAP.md) · [ROLES](./ROLES.md)
