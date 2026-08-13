# bench — R1's operational surface

Everything R1 runs is one command. A sweep that needs a checklist gets run
inconsistently, and two runs that differ in a step nobody wrote down are two
experiments sharing a `run_id`.

Role doc: [guru.md](../docs/guru.md) · Contracts: [CONTRIBUTING](../docs/CONTRIBUTING.md)

```sh
pip install -e .
orch-workers --help
```

Every subcommand also runs as `python -m orchestrator.workers.cli`.

---

## The distinction everything else depends on

**Sweeps and serving are different modes. They are never conflated.**

| | Sweep | Characterization / serving |
|---|---|---|
| Interface | vLLM offline `LLM.generate` | vLLM OpenAI-compatible server |
| Tuned for | Throughput | Latency at fixed concurrency |
| Batch | As large as memory allows | Declared: `--concurrency 1 8` |
| Wall-clock is | **Meaningless as latency** | The measurement |
| Row carries | `mode="sweep"` | `mode="serving"` |

A batched sweep runs hundreds of requests concurrently, so any one request's
wall-clock is a function of queue depth, not of the model. `wall_ms` is
recorded anyway — a sudden change in it is a useful signal that the serving
stack is unwell — but it is **not** the reported latency metric.

This is enforced, not documented: `orch-workers characterize` refuses a
sweep-mode backend, and `orch-workers validate` refuses to regress against
sweep rows. Both refusals name the reason.

---

## Backends

One interface, four implementations. Which one ran is recorded on every row.

| Backend | Mode | Use |
|---|---|---|
| `vllm_offline` | sweep | The sweep engine. Batched `LLM.generate`. |
| `vllm_openai` | serving | Characterization and online serving. |
| `anthropic` | serving | Runs before any GPU exists. **Cannot honour a seed.** |
| `mock` | sweep or serving | Deterministic, offline. CI, and R3/R4's fixtures. |

Arms name a **role** (`small` / `large`), never a model id, so re-running the
whole sweep against a different pair is `--small` / `--large` rather than a
code change. The resolved `model_id` lands on every row.

> `anthropic` sets `honors_seed = False`: the Messages API has no seed
> parameter, so at temperature 0 every seed returns identical text. A
> three-seed frozen ladder would pay three times for one distinct generation.
> The sweep warns rather than letting it surface as a strange confidence
> interval in week 4.

---

## Running a sweep

```sh
orch-workers sweep \
    --tasks data/tasks/mbpp.jsonl \
    --splits data/splits/splits.json \
    --backend vllm_offline \
    --small Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --large Qwen/Qwen2.5-Coder-32B-Instruct \
    --arms direct_small direct_large \
    --seeds 0 1 2 \
    --batch-size 256 \
    --dataset mbpp+
```

**Resumable.** Interrupt it and run the identical command again. Resume is
keyed on `(task_id, arm, seed, params_hash)`: a completed cell is never
recomputed, and a cell computed under different parameters is never mistaken
for a hit. Errors are retried; refusals and truncations are not, because
retrying a real outcome until it changes selects for the lucky sample and
biases the arm's measured accuracy upward.

**Append-only.** Rows are appended, partitioned by `run_id`, never mutated.

**A new experiment is automatically a new run.** Each arm's `params_hash` and
a fingerprint of the corpus are folded into the config the `run_id` hashes, so
editing a prompt template or regenerating the task manifest produces a new run
directory rather than silently extending the old one. Batch size and output
paths are excluded — they change how a sweep runs, not what it produces.

Exit code is non-zero if the truncation rate breaches `--truncation-alarm`.

---

## Characterization — measure once, impute everywhere

```sh
# with a vLLM server already up
orch-workers characterize \
    --backend vllm_openai \
    --backend-option base_url=http://localhost:8000/v1 \
    --small Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --large Qwen/Qwen2.5-Coder-32B-Instruct \
    --concurrency 1 8 \
    --hardware "A100-80GB PCIe" \
    --usd-per-gpu-hour 1.10
```

Writes `bench/cost_coefficients.json` — a measured artifact, committed to the
repo next to the code that produced it.

The probe set is a **grid over prefill length and `max_tokens`**, not a sample
of real tasks. Real prompts correlate length with content; if prefill and
decode move together the two coefficients are not separately identifiable, and
the fit refuses rather than emitting numbers that look plausible.

Three tiers of cost, in descending order of authority:

1. **Tokens** — primary, hardware-independent, already on every row.
2. **GPU-seconds** — `prefill x a + decode x b`.
3. **USD** — GPU-seconds x the instance rate, recorded as a stated assumption.

**GPU-seconds excludes the intercept; imputed latency includes it.** The
intercept is per-request overhead — scheduling, HTTP, detokenization — that a
caller waits for but the GPU does not spend. Charging it to GPU-seconds would
inflate the apparent cost of short generations, which is exactly the cheap arm
the routing question is about.

---

## Re-characterizing does not mean regenerating

Cost lands in a **sidecar**, keyed by a fingerprint of the coefficients:

```
runs/{run_id}/cost/{fingerprint}.jsonl    rollout_id -> gpu_seconds, latency, usd
runs/{run_id}/cost/{fingerprint}.json     which coefficients, and their fit quality
```

```sh
orch-workers impute --run 2026-08-14-a3f91c2-7d4e08
```

A sealed run is immutable — its manifest carries per-file checksums — so
rewriting rows to add two columns would invalidate every number already
computed from it. Two fingerprints coexist happily: the A100 costing and the
H100 costing of the same generations. Consumers pin one, exactly as they pin a
`run_id`.

---

## The definition-of-done check

```sh
orch-workers validate --run 2026-08-14-a3f91c2-7d4e08
```

Regresses imputed latency against measured wall-clock, per model, and requires
**R² > 0.9**. This is the honesty check on the whole cost model: if it is low,
the imputation is fiction and that must be said out loud before R4 builds a
frontier on top of it. Exit code is non-zero when it fails.

Only `mode == "serving"` rows with exact token counts, at the reference
concurrency, are eligible. Point it at a sweep and it refuses.

---

## Run layout

```
runs/{run_id}/
    _CONFIG.json                      written when the run opens
    generations/part-{shard}.jsonl    append-only, one file per writing process
    generations/generations.parquet   written at seal, when pyarrow is present
    cost/{fingerprint}.jsonl          imputed cost, additive, never in-place
    _MANIFEST.json                    write-then-seal; its arrival makes the run valid
```

`run_id = {date}-{git_sha7}-{config_hash6}`. A dirty worktree — **including
untracked files**, because an untracked module changes what a sweep does —
stamps `-dirty` and is non-publishable.

**A run is invalid until `_MANIFEST.json` lands.** Readers skip unsealed runs
by default, which is what stops an interrupted sweep being read as a complete
one: a real number computed over two-thirds of a corpus. Only the resume index
reads unsealed runs, because a resumable run is by definition one that never
got its manifest.

JSONL is authoritative; Parquet is derived. A sweep must not fail at hour nine
because an optional dependency is missing on the GPU box.

```sh
orch-workers runs                 # sealed runs only, as readers see them
orch-workers runs --all           # including unsealed
orch-workers show --run <run_id>  # counts, truncation rate, warnings
```

---

## Code extraction is R1's

R2 receives `code`, never raw model output. If extraction lived in the grader,
changing a prompt template would break R2's tests and R1 would own a bug in
someone else's file.

The hard case is not finding a fence — it is **choosing among several**. Models
routinely emit the implementation and then a usage example, or a first attempt
and then a correction. Picking the longest block, the obvious heuristic, picks
wrong whenever the example is chatty. Candidates are scored on whether they
parse, whether they define the task's entrypoint, and only then on length.
Import-only blocks are merged into the winner, so a solution split as "imports
here, implementation there" does not fail at import time in the sandbox.

Every row records the `extract_strategy` that produced it, so extraction
quality is measurable rather than assumed. A run whose `bare_*` rate moves is a
prompt regression, not a model capability gap.

```sh
orch-workers extract response.txt --entrypoint add   # debug a prompt
```

---

## Tests

```sh
pytest bench/tests
```

No GPU, no network, no API key, a few seconds. The sweep runner, the store,
resume, sealing, and the cost model are exercised end to end on every commit —
a pipeline whose only integration test needs a GPU is one that gets tested when
someone remembers.

The `mock` backend plants a **signal of known strength**: a latent per-task
difficulty both rungs are scored against, so the large model solves a
correlated superset of what the small one solves, with genuine discordance at
the margin. The planted gap clears Phase 0's `A_large − A_small ≥ 8pp`. R3 and
R4 can develop against schema-valid rows whose right answer is known by
construction, without waiting for hardware.
