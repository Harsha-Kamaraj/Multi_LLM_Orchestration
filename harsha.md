# R3 — Policy & Learning · Harsha

You build the thing the project is named after. You also never touch a GPU, never
run a container, and never see the test split.

**Your paths:** `src/orchestrator/policy/`
**Your commit scope:** `policy`
**You block:** R4 (needs your `decisions.parquet`)
**You are blocked by:** nobody — you develop against synthetic fixtures from day 1

---

## Mandate

Given a rollout store, learn to choose an arm that beats the verifier-gated cascade
on the cost–accuracy frontier.

Your input is **the rollout store (Parquet) and nothing else.** If you ever need to
call a model or run a grader to do your job, an interface has been violated —
raise it rather than working around it.

---

## Do not train a scalarized reward

The obvious move is to fit one model on `accuracy − λ·cost − λ₂·latency` and be done.
Don't. It bakes λ into the weights, so every new cost/latency tradeoff is a retrain,
and you can't produce a frontier — only a point.

Learn **three separate heads**:

```
P_pass[a](x)     — calibrated probability arm a solves the task
E_cost[a](x)     — expected cost of arm a
E_latency[a](x)  — expected latency of arm a
```

Then sweep λ at **decision time**:

```python
# λ is a runtime knob, not a training hyperparameter
def decide(x, lam):
    return argmax_a(P_pass[a](x) - lam * E_cost[a](x))
```

One training run, an entire frontier, and λ becomes a product dial rather than an
ML decision. This is also what lets R4 compare against baselines at matched cost.

---

## D0 vs D1 — the asymmetry is the finding

| | D0 (pre-generation) | D1 (post-generation) |
|---|---|---|
| Features | Prompt only | Prompt + candidate code + self-consistency + visible-test outcome |
| Expected AUC | 0.60 – 0.68 | 0.80 – 0.90 |
| Decision | Which arm to try | Whether to escalate |

Observing failure beats predicting it. **Expect D0 to be weak** — that isn't your
model underperforming, it's the actual structure of the problem, and it's the most
interesting number in the project. Report it plainly; don't tune D0 until it looks
respectable.

If `AUC_D1 < 0.75`, the project's premise is false. That's a hard stop, and you're
the one who finds out.

---

## Leakage — you will do this accidentally

Every feature must be **computable at the decision point it claims to serve.**

The obvious leak is using hidden-test outcomes. You won't do that. The leaks that
actually happen:

- A `difficulty` column that was itself derived from pass rates
- Task-level statistics computed over the full dataset before splitting
- Anything aggregated across seeds of the *same* task at inference time
- A feature from ladder step *k+1* used at step *k*
- Normalization constants fit on train+test

Write the feature builder so each feature declares its decision point, and assert it.
R4 runs an independent leakage audit and will find what you missed — that's the
design, not a criticism. Make their job boring.

---

## Calibration is not optional

`P(pass)` must be a probability, not a score. The decision rule subtracts
`lam * E_cost` from it — if `P_pass` is uncalibrated, that subtraction is comparing
incommensurable quantities and λ means nothing.

Fit **isotonic regression on the validation split**, never on test. Target **ECE < 0.05**.
Report reliability diagrams per arm.

---

## Fixtures first — you start on day 1

The synthetic rollout generator emits schema-valid rows with a **planted signal of
known strength**. Build the entire pipeline against it before real data exists.

This is not a stopgap. It's your correctness test: **the right answer is known by
construction**, so a policy that fails to recover the planted signal has a bug, and
you find out in week 1 instead of week 4 when the real store is noisy and you can't
tell a bug from a null result.

Include adversarial fixtures — nulls, unicode, duplicate `task_id`s, missing arms,
mixed `schema_version`.

**The real store lands week 2. Switching to it should be a path change and nothing
else.** If it requires code changes, the contract was violated — say so.

---

## Interface you must satisfy

```
Input:   rollout store (Parquet), partitioned by run_id
Output:  policy.pkl + decisions.parquet, one per (policy, λ)
```

`decisions.parquet` pins the exact `run_id` it was built from. Nothing reads "latest".

---

## Week 1 (Phase 0)

- [ ] Sign off on `schemas/` by day 3
- [ ] Synthetic rollout generator with planted signal + adversarial fixtures
- [ ] D0 and D1 feature builders, each feature declaring its decision point
- [ ] Recover the planted signal end to end
- [ ] Measure `AUC_D0` and `AUC_D1` on the pilot the moment R2 grades it

---

## Definition of done

The policy recovers the planted signal in synthetic rollouts, and `P(pass)` is
calibrated to **ECE < 0.05**.

---

## Keep it boring

Logistic regression and LightGBM on CPU. No RL framework, no fine-tuned policy LM,
no vector DB in Phase 1.

You are choosing among a handful of arms with a few dozen features and ~1,000 tasks.
Using 7B parameters for that is the wrong tool until a measured gap says the tabular
policy can't close it. "We tried the simple thing first and here's the number" is a
stronger result than a complicated one that can't be attributed.

---

## What you must not do

- Open the test split — R4 only, exactly once
- Fit anything, including normalization constants, on test
- Use a hidden-test outcome as a feature, directly or transitively
- Bake λ into training
- Report an uncalibrated probability

---

[README](./README.md) · [CONTRIBUTING](./CONTRIBUTING.md) · [ROADMAP](./ROADMAP.md) · [ROLES](./ROLES.md)
