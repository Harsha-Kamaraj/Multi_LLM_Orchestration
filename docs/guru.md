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

## Your hardware, and what it decides for you

**2× RTX 4090, 24 GB each. Nothing always-on.**

These are consumer cards — **no NVLink, P2P disabled.** Tensor parallelism across
them over PCIe is slow and adds a failure mode you would spend days debugging
instead of building the project. So:

> **Hard rule: every model fits on one card. TP=1, always.**

That rule picks the models:

| Model | bf16 weights | One 4090? |
|---|---|---|
| Qwen2.5-Coder-1.5B — **small arm** | ~3.1 GB | yes |
| Qwen2.5-Coder-7B — **large arm** | ~15.2 GB | yes, ~8 GB for KV |
| Qwen2.5-Coder-14B | ~29.4 GB | no — AWQ only |
| Qwen2.5-Coder-32B | ~65.6 GB | no — AWQ ~19 GB, tight |

**Both arms are bf16, deliberately.** Quantizing only the large arm confounds model
scale with quantization damage, and every accuracy-gap number in the report stops
meaning what it claims. If you ever need a bigger large-arm, quantize *both* or
quantize neither.

**The second card buys throughput, not size.** Small arm sweeps on GPU0, large arm
sweeps on GPU1 — two independent single-GPU vLLM processes, no coordination between
them, roughly half the wall-time. Set `CUDA_VISIBLE_DEVICES` per process and let
them run.

---

## Session-based execution

Nothing runs always-on. That is a constraint, and it has exactly one operational
consequence you must not get wrong:

> **One vLLM process per model, per sweep. Never per task.**

vLLM startup — weight load plus CUDA graph capture — is 30–90 s. Paying that per
task would make a sweep 50× longer and the latency numbers meaningless. Load once,
sweep the entire arm, tear down, load the next.

This is also why the whole project works on your hardware: **log once, replay many
times** means no two models are ever resident simultaneously and nothing downstream
needs a live server.

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

## Latency is two numbers, never one

Because nothing is always-on, "latency" is ambiguous unless you split it. You own
both halves:

| Metric | What it is | Where it goes |
|---|---|---|
| `warm_latency_s` | Steady state, model resident, batch=1 | The routing objective — what the policy optimizes |
| `cold_start_s` | vLLM startup, 30–90 s, measured once per model | Reported **separately**, excluded from the objective |

Cold start is excluded on purpose, and you should be able to say why: any real
deployment amortizes it across many requests, so folding it into per-task latency
would make both arms look identical and hide the actual routing tradeoff. Report it
as a fixed deployment cost instead of burying it.

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

## Status — 13 Aug 2026

**Every line of code R1 owes is written and tested. Not one of it has touched a
GPU.** That is the whole summary, and the tables below are the unflattering
version of it. The distinction matters: a tested code path is not a measurement,
and none of my numbers exist yet.

### Week 1 (Phase 0)

| # | Assigned | Done | Where it actually stands |
|---|---|---|---|
| 1 | Sign off on `schemas/` by day 3 | ✅ | The package landed (R4, 19 commits) and `schemas/tests/test_conformance.py` binds my row shape to it directly — version, required fields, `finish_reason` and `mode` vocabularies, and `rollout_id` derivation are all asserted against `generation.py`. **My row shape is now ratified by a test rather than by agreement.** It landed past day 3 and without a four-role sign-off, so the process clause failed even though the artifact is right. |
| 2 | Sweep runner, rollout store, resume, cost model | ✅ | End to end, 270 tests green in ~23 s, no GPU and no network |
| 3 | vLLM at TP=1, 1.5B on GPU0 / 7B on GPU1 | ❌ | `vllm_offline` and `vllm_openai` backends are written and exercised against `mock`. Neither has been pointed at a real vLLM process. |
| 4 | 200-task pilot sweep → R2 for grading | ❌ | **No longer blocked.** R2 shipped `data/tasks/pilot_200.jsonl` on 13 Aug — 200 tasks, hashed manifest — and `corpus.py` reads it. My only cross-role dependency is closed, so what stands between here and a pilot is GPU access and nothing else. |
| 5 | First characterization pass → `cost_coefficients.json` | ❌ | `characterize.py` and the coefficient fit are written and tested. There is no `bench/cost_coefficients.json`. The coefficients do not exist. |
| 6 | Measure `cold_start_s` for both models | ❌ | Needs a real vLLM startup to time. Nothing to report. |

### Standing responsibilities (ROLES.md)

Split into two columns on purpose — collapsing them is how "built" gets read as
"working".

| Responsibility | Built | Measured on real hardware |
|---|---|---|
| vLLM offline batch sweeps | ✅ | ❌ |
| vLLM OpenAI server for characterization | ✅ | ❌ |
| Arm implementations | ✅ | ❌ |
| Prompt templates, versioned and hashed | ✅ | ❌ |
| Resumable one-command sweep runner | ✅ | ❌ |
| Cost/latency characterization pass | ✅ | ❌ |
| Code extraction (mine, not R2's) | ✅ | ❌ |

Seven arms are registered, not two: `direct_small` / `direct_large`, their
`_notests` variants, `probe_small`, and `repair_small` / `repair_large`. The
last three are Phase 2 deliverables built early — they cost nothing to carry and
they stop the arm registry being reshaped mid-project.

### Definition of done

| Clause | Done | |
|---|---|---|
| A sweep is one command | ✅ | `orch-workers sweep` |
| It is resumable | ✅ | Keyed on `(task_id, arm, seed, params_hash)`, tested |
| Imputed latency correlates with wall-clock at **R² > 0.9** | ❌ | The check is implemented — `orch-workers validate`, non-zero exit below 0.9. It has never run against real data, so the R² is not low, it is **absent**. |

R1 is not done. Two of three clauses are code properties I can prove today; the
third is a measurement, and it is the one the rest of the project leans on.

### What unblocks what

Nothing downstream of me is waiting on code, and as of 13 Aug nothing upstream
of me is missing either. R2's manifest landed; **GPU access is now the single
remaining blocker** on rows 3–6, and it is the one item on this page that no
amount of my own work can clear. The pilot sweep is a day's work once a card is
available, because the pipeline it runs through has been tested on every commit
since it was written.

Phase 0's number is `A_large − A_small ≥ 8pp`. Expect ~20pp from 1.5B vs 7B. If
the arms aren't differentiated, **shrink the small model** — drop to 0.5B before
you reach for a bigger large-arm, because growing the large arm costs you TP=1.

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
- Reach for tensor parallelism across the two 4090s
- Quantize one arm and not the other
- Load a model per task instead of per sweep
- Fold `cold_start_s` into per-task latency
- Change a prompt template without changing `params_hash`
- Mutate a row in the rollout store
- Grade anything — that's R2's, and the separation is what makes grading trustworthy
- Open the test split — R4 only

---

## What's built

`src/orchestrator/workers/` and `bench/`. Operational detail lives in
[bench/README.md](../bench/README.md); this is the map and the reasoning.

| Module | Holds |
|---|---|
| `params.py` | `GenParams` and `params_hash` — row identity |
| `prompts.py` | Versioned templates, hashed by text; the label is not the identity |
| `arms.py` | Arm registry. Arms name a *role*, never a model id |
| `extract.py` | Code extraction — mine, not R2's |
| `backends/` | One interface: `vllm_offline`, `vllm_openai`, `anthropic`, `mock` |
| `generation.py` | The `Generation` record and its rollout-row projection |
| `runid.py` | `{date}-{git_sha7}-{config_hash6}`, dirty stamping |
| `store.py` | Append-only, write-then-seal, checksummed per part |
| `resume.py` | Keyed on `(task_id, arm, seed, params_hash)` |
| `corpus.py` | Reads R2's manifest and R4's splits; owns neither |
| `sweep.py` | The one command |
| `cost.py` | Coefficients, fitting, and the R² definition-of-done check |
| `characterize.py` | The grid probe pass, serving-mode only |
| `impute.py` | Cost sidecars — re-costing never rewrites a row |
| `cli.py` | `orch-workers` |

270 tests in `bench/tests`, no GPU and no network, ~23 s. The pipeline is
exercised end to end on every commit rather than when someone remembers — though
"on every commit" is currently a local habit, not a CI job: see
[CONTRIBUTING](./CONTRIBUTING.md) → *What is not enforced yet*.

### Decisions worth arguing with

**Backends are pluggable and the substrate is a flag.** The role doc says vLLM;
the scaffold was Anthropic-only. Both now sit behind one interface, so "the GPU
landed" is a `--backend` change and nothing downstream of me moves. It also
means CI runs the real sweep path against a deterministic mock.

**Sweep-versus-serving is enforced, not documented.** Every row carries `mode`
and `batch_size`. `characterize` refuses a sweep-mode backend; `validate`
refuses to regress against sweep rows. Both refusals say why. If someone
removes those two checks, every latency number in the project silently becomes
a queue-depth measurement, and it will look like plausible data.

**The corpus content is in the `run_id`, not just its path.** If R2 regenerates
`data/tasks/*.jsonl` in place, that is a different experiment and now gets a
different run directory automatically.

**Untracked files count as dirty.** An untracked module inside the package
changes what a sweep does while leaving the recorded sha pointing at code that
never ran.

**Errors are retried on resume; refusals and truncations are not.** A timeout
deserves another attempt. Retrying a real outcome until it changes selects for
the lucky sample and biases the arm's accuracy upward.

**Cost is a sidecar, not a column.** Re-characterizing on new hardware
re-costs every existing rollout without regenerating a token — which is the
thing the three-tier cost model is *for*. Two costings of one run coexist.

### Open, and not mine to decide alone

- **Do I generate on the test split?** I do today — R4 cannot evaluate rollouts
  that do not exist, and I never see a grade, so nothing is "opened". Raising
  it because my own don't-do list says the words. `--include-splits` restricts
  it if the team disagrees.
- **Seeds under a hosted backend.** `anthropic` has no seed parameter, so at
  temperature 0 the three-seed ladder buys one distinct generation. The sweep
  warns. If we run the pilot on a hosted API, use one seed or raise temperature.
- **`graders/base.extract_code` is now redundant** — extraction is mine and R2
  receives `code`. R2's call whether to drop it.

---

[README](../README.md) · [CONTRIBUTING](./CONTRIBUTING.md) · [ROADMAP](./ROADMAP.md) · [ROLES](./ROLES.md) · [bench](../bench/README.md)
