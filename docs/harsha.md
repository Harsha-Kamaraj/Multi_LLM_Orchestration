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

## You are measured against a heuristic, and you don't build it

R4 owns a `heuristic_route` baseline: hand-specified prompt-only rules with
thresholds tuned on validation and swept across λ. It sits directly below
`learned_D0` on the capacity ladder, with the **same information** and far fewer
free parameters.

| Rung | Free parameters | Information |
|---|---|---|
| `heuristic_route` (R4) | 1–2 thresholds | prompt only |
| `learned_D0` (you) | dozens | prompt only |
| `learned_D1` (you) | dozens | prompt + code + visible tests |

That adjacency is the point: `heuristic_route → learned_D0` isolates whether
**learning** adds anything over human priors, with information held constant.

**Do not build it, and do not tune it.** Same reasoning as the test split — the
person whose policy is being compared shouldn't control its competition. If it beats
you, that's a real result and it gets reported as one.

What it does mean for you: **`learned_D0` has to justify dozens of parameters
against two.** If your D0 model only ties a prompt-length threshold, say so plainly
rather than adding features until it wins.

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

**All five are closed in code, each structurally rather than by rule** —
`_synth_*` quarantined into `RolloutData.latent`, normalization fitted on train
only and refusing `test`, siblings excluded by identity, and ladder steps
truncated away rather than merely forbidden. The pattern throughout: a feature
builder is handed an object that does not *contain* what it must not read.

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

## Status — 14 Aug 2026

Phases 2–4 merged as `10a512f`, Phase 5 as `3bbb569`, Phase 6 as `9b7374d`,
Phase 7 as `d096275`. 983 tests green, no GPU and no network.

**R3's code is complete.** Everything below is measured on synthetic fixtures:
no number here has touched the real pilot, and that is now the only thing
standing between this project and its Phase 0 verdict.

| | Item | Where it stands |
|---|---|---|
| ✅ | Synthetic rollout generator with planted signal + adversarial fixtures | Delivered by R4 as `schemas/synth.py` and `schemas/adversarial.py`, so nothing blocked the start |
| ✅ | A reader that cannot leak | `store.py` returns rows with the hidden-test columns **physically removed** and hands labels back separately, keyed by `rollout_id`. A feature builder is never given the object holding them. The test split has no load flag at all. Every read pins an explicit `run_id`. |
| ✅ | The row contract | `contract.py` — `observable_at(decision_point)` names which columns exist at D0 versus D1, plus `normalize_row`, `is_graded`, `solved`, and `assert_no_labels(columns, context=…)` |
| ✅ | D0 and D1 feature builders, each feature declaring its decision point | `features/` — 13 at D0, 34 at D1, D0 a verified subset of D1. Declared columns are checked against `observable_at()` at construction, *and* each feature is handed a `RowView` that raises on any undeclared read. The second layer is what makes the first trustworthy. |
| ✅ | Three value heads — `P_pass`, `E_cost`, `E_latency` | `heads.py`, fitted per arm. `E_cost` predicts **prefill and decode tokens**; conversion to GPU-seconds, latency and USD happens at scoring time through R1's pinned `CostCoefficients`, so a change of instance price is not a retrain. |
| ✅ | Calibration to ECE < 0.05 | `calibration.py`. Isotonic on `val`, ECE measured by cross-fitting with folds **grouped by task**. See the note below: at pilot scale isotonic is often measurably worse than doing nothing, and the code records that decision rather than hiding it. |
| ✅ | The λ-sweep and `decisions.parquet` | `decide.py` — `argmax_a [P_pass − λ·E_cost]` over a frozen 121-point log grid, written as Parquet with JSONL as the authoritative copy. Verified against R4's own code: decisions replayed through `eval.policies.from_decisions` run, and at the degenerate ends reproduce `always_large` and `always_small` **exactly**, which is the check that the cost accounting agrees. |
| ✅ | Recover the planted signal end to end | On a 500-task fixture: `AUC_D0 = 0.6464 [0.5485, 0.7384]`, `AUC_D1 = 0.8584 [0.8009, 0.9086]`, gap `+0.2120`. Both land inside ROADMAP's predicted bands, and nothing is tuned. |
| ✅ | Repair ladder, and the leak that lives in it | `ladder.py` — chains assembled by walking `parent_rollout_id` to a root, cost cumulative along the path. **This closes the fifth of the five leaks named below**, structurally: `Ladder.upto(k)` returns a chain that does not *contain* the later steps, so a step-k caller cannot read step k+1 however hard it tries. |
| ✅ | Self-consistency probe, charged | `decide.probe_surcharge` prices k=3 cheap draws and adds them to every arm. A test proves it does not move the argmax — it is paid before routing, so it shifts the policy rightward on the frontier and has to earn its cost through better decisions rather than free information. See the note below for why a probing router still cannot be run end to end. |
| ❌ | Measure `AUC_D0` and `AUC_D1` on the pilot | **The last two unmeasured quantities in the Phase 0 gate, and both are R3's.** R1 swept `2026-08-13-c76a55d-4f4767` and R2 graded all 1200 rows on 13 Aug, so the data exists — but `runs/` is gitignored, so the graded store is not in the repo and R3 has never read it. `orch-policy gate` runs the moment that directory is available. |
| ❌ | Sign off on `schemas/` by day 3 | Missed, and not recoverable — the contract R3 consumes was frozen without this seat's review. Four consequences are open with R4 and R2: the rollout row carries no prompt, R1's Parquet `extra` does not satisfy `rollout.schema.json`, the `generations/` vs `rollouts/` layer split is not in the schema, and there is no entry point for replaying a **D1** policy (below). |

The three leakage guarantees this package claims are enforced in code rather
than in review, which is the right shape: R4's independent audit
(`eval/leakage.py`) should find nothing, and if it does, that is the design
working.

### Calibration did not do what this document assumed it would

Worth recording, because the definition of done says "isotonic" and the honest
answer at this data size is "isotonic, measured, and usually declined".

Isotonic is nonparametric and needs data. On a few hundred validation rows it
fits the noise in the reliability curve, and cross-fitted ECE comes out *worse*
than leaving the classifier's own probabilities alone — logistic regression
fitted by maximum likelihood is already close to calibrated, so there is little
to correct and plenty to break. Sweeping fixture size puts the crossover at
roughly 1,800 validation rows:

| val rows | uncalibrated ECE | cross-fitted ECE | isotonic kept |
|---|---|---|---|
| 180 | 0.0499 | 0.1103 | no |
| 720 | 0.0276 | 0.0376 | no |
| 1,800 | 0.0319 | **0.0120** | yes |

So both candidates are measured out of fold and the better one ships, recorded
as `applied` in `heads.json`. **The pilot has roughly 120 validation rows per
arm, so expect the identity map to win on real data.** That is a finding about
how much validation data a calibrated policy needs, not a licence to skip
calibration — the measurement is what licenses the choice.

Reporting ECE in-sample would have hidden all of this: isotonic drives the
in-sample number to 0.0000 on any data at all, and a test pins that value
precisely so the cross-fitting cannot quietly stop happening.

### A D1 policy has nowhere to be replayed

`eval.policies.from_decisions` replays one action per task and charges for that
arm. That is exactly a D0 routing decision, and `decisions.parquet` feeds it
directly.

A D1 policy decides something structurally different — *whether to escalate*,
after the cheap arm has already generated. Replaying that as "the action was
`large`" charges for the large arm alone and discards the small arm's cost,
which is already spent and cannot be un-spent. The result would understate the
policy's cost and flatter it against every cascade baseline it is meant to be
compared with.

So `decide.py` **refuses** a D1 policy rather than emitting it in a shape that
lies. Closing this needs a cascade-accounting entry point on R4's side; until
then `learned_D1` cannot enter the comparison, even though its heads are fitted
and its AUC is the strongest number R3 has.

### Repair looks like a better buy than escalation — on invented semantics

ROADMAP's Phase 2 gate asks whether repair pays for itself. On a fixture ladder:

| strategy | accuracy | cost | |
|---|---|---|---|
| `always_small` | 0.2573 | 1.0331 | the floor |
| `repair_on_failure` | 0.4958 | 2.1240 | Δacc/Δcost = **0.219** |
| `escalate_on_failure` | 0.5479 | 5.7145 | Δacc/Δcost = **0.062** |

Escalation buys more accuracy; repair buys it about 3.5× more efficiently, and
wins on utility for 77 of the 121 λ values. All three are charged the cheap
attempt, because all three take it — the choice between repairing and escalating
is only reachable *after* the cheap sample has failed, and charging repair for
the repair alone would compare it against a baseline that never ran.

**The repair success curve is invented.** Nothing has been run through R1's
`repair_small` arm, so `fixtures.add_repair_ladder` plants a plausible
structure — a candidate that passed three of four visible tests is more
repairable than one that passed none — and the numbers above inherit it. The
accounting is real; the result is not evidence about real repair.

One bug worth recording, because it was live in exactly the state the real store
is in today. With **no** repair rows, the repair strategy silently *becomes*
`always_small`, which beats escalation at high λ purely by declining to spend —
and the gate reported `PASS`, "repair pays for itself", having repaired nothing.
There is now a third verdict, `NO REPAIRS`, and it exits non-zero.

### A probing router has no decision point to live at

The probe is bought *before* routing, which makes it a D0 decision. Its features
read agreement across generations that have already happened, which are D1
columns. `feature_set("D0", with_probe=True)` therefore refuses — correctly,
and R3 wrote that refusal in Phase 3 without noticing it would later close this
door.

So the contract's two decision points are one short of the three surfaces this
project actually has: route at D0, route after a paid probe, escalate at D1.
Inventing the third unilaterally is a schema change needing all-four sign-off,
so it is recorded as a test instead.

Separately: `from_decisions` charges the logged cost of the chosen arm and knows
nothing about a probe, so `probe_cost_per_task` is reported in the decisions
manifest for R4 to add. Until they do, a probing policy would be compared at a
cost it never paid.

### The frontier is narrow, and that is the D0 weakness again

The frozen λ grid is 121 points across six decades — twenty per decade, which
looks excessive until you sweep it. The region where tasks route *both* ways
spans about a fifth of a decade. At four points per decade the entire trade-off
collapses to a single λ and the frontier renders as one cliff.

The cause is not the grid. An arm is chosen over a cheaper one when the quality
gap beats λ times the cost gap, so tasks flip at `Δp / Δcost`. A D0 policy that
separates tasks weakly predicts nearly the same `Δp` for all of them, so they
all flip at nearly the same λ. **A narrow frontier is what a weak router looks
like at decision time**, and it is the same finding as `AUC_D0 ≈ 0.65` arriving
through a second route. `Sweep.summary` says so out loud when fewer than three
λ values route both ways, rather than printing a tidy cliff.

---

## Definition of done

The policy recovers the planted signal in synthetic rollouts, and `P(pass)` is
calibrated to **ECE < 0.05**.

**Met on synthetic data as of 14 Aug**, both clauses: the gate recovers the
planted signal inside its predicted band, and every arm clears the ECE target
at D1. It is not met on the pilot, because R3 has not yet been given the graded
store — and a definition of done evaluated only against fixtures is a statement
about the code, not about the claim this project exists to test.

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
- Build or tune `heuristic_route` — that's R4's, by design
- Report an uncalibrated probability

---

[README](../README.md) · [CONTRIBUTING](./CONTRIBUTING.md) · [ROADMAP](./ROADMAP.md) · [ROLES](./ROLES.md)
