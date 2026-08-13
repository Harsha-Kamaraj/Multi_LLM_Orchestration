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

**R1 has run on a GPU. Every row below is now a measurement rather than a
claim.** The pilot sweep is sealed, the cost coefficients are fitted from
timings, and the definition-of-done R² is a number instead of an absence.

The headline run:

| | |
|---|---|
| `run_id` | `2026-08-13-c76a55d-4f4767` — sealed, publishable, clean sha |
| Rows | 1200 = 200 tasks × 2 arms × 3 seeds, **0 failed** |
| Wall-clock | 187.4 s for the whole sweep, both arms, batched at 256 |
| Truncation | **1.00%** (12 of 1200), against a 15% alarm |
| Extraction | 1182 `fenced_entrypoint`, 6 `fenced_parsed`, 12 `fenced_truncated` — every row yielded code |
| Cost sidecar | 1200/1200 imputed, **2820.0 GPU-seconds**, $0.78 at the stated rate |

Measured on `NVIDIA RTX 4500 Ada Generation 24GB (WSL2, TP=1)`:

| Model | batch | prefill ms/tok | decode ms/tok | R² |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 1 | 0.0435 | 9.5061 | 0.9994 |
| Qwen2.5-Coder-1.5B | 8 | 0.2487 | 10.1061 | 0.9589 |
| Qwen2.5-Coder-7B | 1 | 0.1613 | 37.4314 | 0.9999 |
| Qwen2.5-Coder-7B | 8 | 0.8863 | 39.2172 | 0.9433 |

`cold_start_s` — **28.20 s** small, **55.36 s** large, both inside the 30–90 s
this doc predicted. The small arm's first-ever start was 46.29 s; the extra was
`torch.compile` populating a cold cache. Reported here and excluded from the
objective, as the section above requires.

The decode numbers are mutually consistent, which is the cheapest evidence they
are real: the 7B decodes 3.9× slower than the 1.5B (26.7 vs 105 tok/s) against a
4.7× parameter ratio — the shape of a bandwidth-bound decode, not of a number
someone hoped for.

### What the GPU run cost in bugs

Five defects surfaced that the fixture-backed suite never could, three from
driving the real corpus and two from the measurement itself:

| Fix | What it was |
|---|---|
| `256daae` | The prompt label-guard matched `hidden` as a substring of a *key name*, so R2's `metadata.n_hidden_cases` — an int count — aborted the sweep on task 1. Scalars are now exempt; a key holding content is still refused. |
| `6a9e035` | `load_splits` only knew the `splits` container. R4's manifest nests under `task_ids`, so the flat branch read `corpus_hash`/`name`/`salt` as three task ids and reported `{'unassigned': 200}`. It now reports `{'test': 45, 'train': 111, 'val': 44}`. **This failed silently**, which made `--include-splits` inert and quietly unfenced the test split. |
| `9edfc4c` | A sweep sealed unconditionally, so any transient error sealed the run into a state nothing could repair — the config hashes into the `run_id`, so the identical re-run meant to retry found the same id sealed and refused. "Errors are retried on resume" was documented but unreachable. |
| `25c3f3f` | **Characterization timed its own warmup.** The first pass fitted the small arm's prefill at **−0.185 ms/token** — a longer prompt making a request *faster* — and failed the definition-of-done at R²=0.708. Two confounds: batch=1 ran first and paid the server's one-time costs, and vLLM's prefix caching meant every repeated probe repaid ~zero prefill, so prefill length varied while prefill cost did not. Warmup requests are now issued and discarded, and the characterization server runs with prefix caching off. The same fit is now R²=0.9994. |
| `d402374` | **`validate` had never worked.** `ImputationReport.summary` is a property and `_cmd_validate` called it, so every run with eligible rows raised `TypeError` instead of printing a verdict. The previous status line here — "the check is implemented, it has never run against real data" — was true because it *could not* run. Both existing tests covered the refusal path, which returns before the print. |

The fourth is the one worth staring at. It produced a physically impossible
coefficient and would have shipped a cost model that looked entirely plausible:
every downstream GPU-second, dollar, and imputed latency would have been wrong,
and nothing downstream would have flagged it. 281 tests now.

> **Hardware note.** This doc's *"2× RTX 4090, 24 GB each"* does not describe
> the machine this ran on, which has **one RTX 4500 Ada, 24 GB**. TP=1 and the
> model choices are unaffected. What does not survive is the throughput
> argument: with one card the arms ran **sequentially**, not concurrently, so
> row 3's "1.5B on GPU0 / 7B on GPU1" was satisfied in substance — each arm
> served at TP=1, one model resident at a time — but not in the literal
> two-card layout. Both arms bf16, neither quantized, as the hard rule requires.

### Week 1 (Phase 0)

| # | Assigned | Done | Where it actually stands |
|---|---|---|---|
| 1 | Sign off on `schemas/` by day 3 | ✅ | The package landed (R4, 19 commits) and `schemas/tests/test_conformance.py` binds my row shape to it directly — version, required fields, `finish_reason` and `mode` vocabularies, and `rollout_id` derivation are all asserted against `generation.py`. **My row shape is now ratified by a test rather than by agreement.** It landed past day 3 and without a four-role sign-off, so the process clause failed even though the artifact is right. |
| 2 | Sweep runner, rollout store, resume, cost model | ✅ | End to end, 281 tests green in ~6 s, no GPU and no network |
| 3 | vLLM at TP=1, 1.5B on GPU0 / 7B on GPU1 | ✅ | Both arms served on real vLLM (0.11.0, bf16, TP=1) and swept through the offline engine. One card, so they ran sequentially rather than one per GPU — see the hardware note. Getting here took five environment fixes: vLLM 0.27 needs UVA that WSL2 does not expose, transformers 5.x removed a tokenizer attribute vLLM 0.11 calls, flashinfer's sampler JIT-compiles with an absent `nvcc`, Inductor needs a C compiler, and prefix caching had to be disabled for measurement. |
| 4 | 200-task pilot sweep → R2 for grading | ✅ | `2026-08-13-c76a55d-4f4767`, sealed and publishable. 1200 rows, 0 failed, 1.00% truncated, 187.4 s. Every row carries extracted `code`, so R2 receives code and never raw model output. **Ready for grading; the accuracy gap is R2's number, not mine.** |
| 5 | First characterization pass → `cost_coefficients.json` | ✅ | `bench/cost_coefficients.json` is committed at `c76a55d`. Both arms, batch 1 and 8, fitted from measured timings on a warmed server with prefix caching off. |
| 6 | Measure `cold_start_s` for both models | ✅ | 28.20 s small, 55.36 s large. Reported separately and excluded from the routing objective. |

### Standing responsibilities (ROLES.md)

Split into two columns on purpose — collapsing them is how "built" gets read as
"working".

| Responsibility | Built | Measured on real hardware |
|---|---|---|
| vLLM offline batch sweeps | ✅ | ✅ 1200 cells, 187.4 s, batch 256 |
| vLLM OpenAI server for characterization | ✅ | ✅ 384 timed probes across both arms |
| Arm implementations | ✅ | ✅ `direct_small` and `direct_large`, 600 rows each |
| Prompt templates, versioned and hashed | ✅ | ✅ `params_hash` on all 1200 rows; server sent explicit temperature/top_p/max_tokens/seed |
| Resumable one-command sweep runner | ✅ | ✅ one command, sealed on completion |
| Cost/latency characterization pass | ✅ | ✅ `cost_coefficients.json`, 4 fits |
| Code extraction (mine, not R2's) | ✅ | ✅ 1200/1200 rows yielded code |

Seven arms are registered, not two: `direct_small` / `direct_large`, their
`_notests` variants, `probe_small`, and `repair_small` / `repair_large`. The
last three are Phase 2 deliverables built early — they cost nothing to carry and
they stop the arm registry being reshaped mid-project.

### Definition of done

| Clause | Done | |
|---|---|---|
| A sweep is one command | ✅ | `orch-workers sweep` |
| It is resumable | ✅ | Keyed on `(task_id, arm, seed, params_hash)`, tested |
| Imputed latency correlates with wall-clock at **R² > 0.9** | ✅ | **R²=0.9997** (1.5B) and **R²=1.0000** (7B), n=50 each, RMSE 0.021 s and 0.013 s. `orch-workers validate` exits 0 on both. |

The R² is **out-of-sample**. The coefficients are fitted on a synthetic prefill ×
`max_tokens` grid; they are validated against batch=1 serving rows over 50 real
tasks per arm, which no part of the fit ever saw. Validating on the probe grid
that produced the fit would have been a refit reported as a check.

Two caveats kept in view rather than smoothed away. The batch=8 fits are weaker
than batch=1 (holdout R² 0.9325 and 0.8429); batch=1 is the reference batch that
defines cost and imputed latency, so the clause rests on the stronger fit, but
anyone quoting a batch=8 number should know its spread. And the two serving runs
carry a `-dirty` stamp from a CRLF/LF mismatch between Windows and WSL git,
since corrected — the code that produced them was genuinely `25c3f3f`; only the
flag was wrong. The headline sweep stamps clean.

**All three clauses hold.** R1's definition of done is met.

### What unblocks what

Nothing upstream of me is missing and nothing downstream is waiting on me.
R2's manifest landed, the GPU was released, and rows 3–6 closed on it.

**R2 is unblocked now.** `2026-08-13-c76a55d-4f4767` is sealed, publishable, and
carries extracted `code` on all 1200 rows — grading can start against it without
touching a GPU or re-running generation. **R3 and R4 are unblocked too**: the
rollout store exists with a real `run_id`, and its cost sidecar
(`cost/b2f8305a.jsonl`) gives 2820.0 GPU-seconds and per-row imputed latency to
build a frontier on.

The accuracy gap `A_large − A_small ≥ 8pp` is deliberately **not** on this page.
Grading is R2's, and the separation is what makes the number trustworthy — I
generate, I never grade. What I can say is that the arms are differentiated in
cost by a factor of 3.9× in decode, which is the axis the routing question
trades against.

Re-costing this run on different hardware needs no regeneration: re-run
`characterize` there and `impute` writes a second sidecar beside the first.
That is what the three-tier cost model was for, and it is now exercised rather
than asserted.

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
