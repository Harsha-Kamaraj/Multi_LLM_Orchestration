# R4 — Evaluation & Analysis · Vivian

You are the referee. Your job is to be the hardest person on this team to convince,
including when the result is good.

**Your paths:** `eval/`, `data/splits/`, `schemas/`
**Your commit scopes:** `eval`, `splits`, `schemas`
**You block:** nobody
**You are blocked by:** nobody — you develop against synthetic fixtures from day 1

---

## Mandate

Determine whether the policy actually beat the baselines, and produce a report whose
every number carries an interval and can be reproduced from a `run_id`.

If the answer is "no", that is a successful outcome of your role.

---

## You own the test split

You are the **only** role permitted to open it. It is opened **exactly once**,
after the analysis is pre-registered.

This is why R3 and R4 are different people. The engineer who trains the policy must
not be the engineer who proves it won — it removes the incentive to tune against
test, and it means the leakage audit is run by someone with no stake in passing it.
On a four-person team this is the cheapest integrity control available. It costs
nothing but the discipline to not merge the roles when the schedule tightens.

**Pre-register before unblinding:** the metric, the comparison, the test, the
correction, the stopping rule. Write it down, commit it, then look.

---

## The baseline that matters

`verifier_gated_cascade` — run small, execute the visible tests, escalate on failure.

This is the real competition, and it is embarrassingly strong, because observing
failure beats predicting it. **Any framing that omits it is dishonest.** A routing
policy that beats `always_small` and `always_large` but loses to the cascade has
demonstrated nothing.

All seven, all required:

| Baseline | Why it's in the set |
|---|---|
| `always_small` | Cost floor |
| `always_large` | Accuracy ceiling for a single arm |
| `random_route(p)` | Controls for "any routing at all helps" |
| `heuristic_route` | Controls for "learning helps" — see below |
| `best_of_n_small` | Controls for "more samples helps" — matched cost |
| `verifier_gated_cascade` | **The one to beat** |
| `oracle_router` | Headroom; the gap that's theoretically available |

Compare **at matched cost**, on the frontier. A win at a different price point is
not a win.

---

## `heuristic_route` — the ablation of learning

This is yours, and it is the baseline that answers the question an interviewer asks
first: *"couldn't you have just used prompt length?"*

Arrange the set as a **capacity ladder** and each rung isolates one variable:

| Rung | Free parameters | Information |
|---|---|---|
| `random_route(p)` | 0 | none |
| `heuristic_route` | 1–2 thresholds | prompt only |
| `learned_D0` | dozens | prompt only |
| `learned_D1` | dozens | prompt + code + visible tests |

Two adjacent comparisons carry the result:

- `heuristic_route` → `learned_D0` isolates **learning**, information held constant
- `learned_D0` → `learned_D1` isolates **information**, learning held constant

### Tune it as hard as you tune the policy

This is the whole ballgame. An untuned heuristic compared against a tuned model is
not a baseline, it's a strawman — and it's the most common way "we compared against
baselines" quietly becomes dishonest.

- Fit thresholds on the **validation split** — the same split the policy uses
- **Sweep across λ to produce a frontier**, never a single point. A curve compared
  to a dot is not a comparison
- Try several heuristic families, report the **best**, not the first

**If the tuned heuristic beats the learned policy, that is the finding.** The policy
didn't earn its complexity. Report it as the headline — you are the only person on
this team positioned to say it.

### Keep it a heuristic

Hand-specified rules with tuned thresholds. **Not** a fitted model over prompt
features — the moment you fit logistic regression on prompt signals you have
rebuilt `learned_D0` and the comparison collapses to noise.

Candidate signals, all prompt-only, all cheap:

- prompt token count
- number of visible test cases
- nested-structure and loop-keyword counts in the signature and docstring
- algorithmic keywords: `graph`, `optimize`, `dynamic`, `recursive`

> **Trap:** never use a dataset-provided difficulty label that was derived from model
> pass rates. That is leakage in disguise, and it makes the heuristic look
> artificially strong.

This costs no GPU time — it is a pure function over the rollout store you already
have. It is also a new family in the comparison set, so it enters the
Benjamini–Hochberg correction alongside the λ sweep.

> Closest published reference is **RouteLLM** (LMSYS, 2024). Don't cite anything you
> can't describe precisely.

---

## Statistics

**Paired designs.** Every arm ran on every task. Use that — paired comparisons on
the same tasks, never two independent means.

**Cluster bootstrap**, 10,000 resamples, resampling **tasks *and* seeds**. Naive
bootstrap over rows treats 3 seeds of one task as 3 independent observations and
produces intervals that are far too narrow. This is the single easiest way to ship
a confidently wrong result.

**McNemar exact** for paired binary outcomes.

**Benjamini–Hochberg** across the λ sweep. You are testing many points on a frontier;
uncorrected, something will look significant.

**Power analysis up front** — roughly 1,000–1,500 tasks for a 3pp effect at 80% power,
computed off measured discordance rates, not guessed ones. Do this in week 1, before
anyone commits to a corpus size.

> **No bare means in the report.** Every number carries an interval. This is a merge
> blocker, and it applies to your own numbers first.

---

## The cascade's weakness is p95, not the mean

Escalation is serial: on escalated tasks, latency ≈ small + large. The mean hides
this; **p95 is where a router can genuinely beat the cascade** by skipping the doomed
first attempt.

Report the full latency distribution. A mean-only latency comparison quietly favors
the cascade and hides your most likely real win.

---

## Router confusion matrix

Accuracy alone can't tell you *why* a router won or lost. Report all five:

| Outcome | Meaning |
|---|---|
| `correct_small` | Routed cheap, solved — the win condition |
| `false_escalation` | Paid for large, small would have sufficed — wasted money |
| `missed_escalation` | Stayed small, failed, large would have solved it — lost accuracy |
| `correct_escalation` | Escalated and large solved it |
| `unsolvable` | Neither arm solves it — not a routing failure, don't count it as one |

`unsolvable` matters: it's the ceiling. A router can't be blamed for tasks no arm
solves, and folding them into the error rate makes every router look worse than it is.

---

## Reproducibility

**`run_id` = `{date}-{git_sha7}-{config_hash6}`.** Dirty worktrees stamp `-dirty`
and are **non-publishable**.

**Write-then-seal.** A run directory is invalid until `_MANIFEST.json` lands. Readers
skip unsealed runs. Never overwrite a run — a re-grade is a new `run_id`.

**Nothing reads "latest".** Every consumer pins an explicit `run_id`.

**Golden-run test.** Re-running the pipeline reproduces `results.json` byte-identically.
A change that alters it must be marked `BREAKING-GOLDEN:` in the PR body with the
diff explained. This is your regression gate, and it is the thing that catches a
silent statistical bug three weeks after it lands.

---

## Fixtures first — you start on day 1

The synthetic generator plants a **known-optimal policy**. That means you can test
the *evaluator itself*: your stats code should conclude the planted-optimal policy
wins, at the planted effect size, with an interval that covers it.

An evaluation harness that has never been run against a known answer is not validated.

---

## Week 1 (Phase 0) — status at 13 Aug 2026

R4 is the furthest-along seat: `schemas/` and `eval/` both landed, 209 tests
green. What is missing is not code, it is anything to point the code at.

| | Item | Where it stands |
|---|---|---|
| ✅ | Own the `schemas/` freeze | `schemas/` is frozen — `Task` + `Rollout` JSON schemas, `SCHEMA_VERSION` guards, a validation layer with rollout invariants, and a conformance test binding R1's emitted row to the contract |
| ❌ | …day 3, all four sign off | Landed unilaterally and past day 3. The artifact is right; the ratification step named in `docs/CONTRIBUTING.md` § Schema changes did not happen. |
| ❌ | Power analysis off measured discordance → the corpus size everyone commits to | `mcnemar_sample_size()` and `bootstrap_power()` are implemented and tested. Both need a **measured** discordance rate, and there is no pilot. The 1,000–1,500 figure below is still an assumption. |
| ✅ | All seven baselines implemented against fixtures, including `heuristic_route` | Six in `standard_baselines()`; `heuristic_route` lives in `eval/heuristics.py` because it must be fitted on validation first. It is tuned across families and swept to a frontier via `tuned_frontier()`, not shipped as a point. |
| ✅ | Cluster bootstrap + McNemar, validated on the planted signal | `cluster_bootstrap` resamples tasks *and* seeds, `mcnemar_exact`, `benjamini_hochberg`. `eval/tests/test_integration.py` checks the planted-optimal policy is recovered at the planted effect size. |
| ❌ | Oracle-gap study on the pilot | `oracle_headroom()` is implemented and tested. There is no pilot. |
| ✅ | Leakage audit tooling — independent of R3's feature code | `eval/leakage.py` — seven independent checks: the planted canary, a column allowlist, split disjointness, an AUC upper bound, normalization scope, seed aggregation, and ladder causality. It imports nothing from R3, which is the property that makes it an audit rather than a self-check. |

Also built, beyond the week-1 list: the matched-cost frontier
(`compare_at_matched_cost`, `pareto_front`, `frontier_dominates`), the full
five-cell router confusion matrix, a loader that refuses the test split unless
`unlock_test_split()` is called with a reason and a committed pre-registration
file, a deterministic `results.json` builder with `compare_to_golden()`, and the
`orch-eval` CLI (`report`, `audit`, `power`, `golden`).

**Still open for R4:**

| | |
|---|---|
| ❌ | `data/splits/` — the 60/20/20 split manifest does not exist |
| ❌ | A **committed golden** `results.json`. `compare_to_golden()` is written and tested; there is no baseline file for it to compare against, so the regression gate has nothing to guard yet. |
| ❌ | A pre-registration document. The loader demands one before unlocking test; none is committed. |
| ❌ | An actual report — the builder is real, the numbers it would format do not exist |

---

## Definition of done

Every reported number carries a confidence interval, and re-running the pipeline
reproduces the report **byte-identically** from a `run_id`.

---

## Report the null

If the routing win exists only on the contaminated split and vanishes on
LiveCodeBench post-cutoff, **that is the finding.** Write it up as the headline.

A rigorously-established negative result is a stronger portfolio artifact than a
positive one nobody can reproduce — and on this team, you are the only person
positioned to say it.

---

## What you must not do

- Open the test split more than once
- Report a mean without an interval
- Bootstrap over rows instead of clustering on tasks and seeds
- Compare at unmatched cost
- Drop `verifier_gated_cascade` from the baseline set
- Ship `heuristic_route` untuned, or as a single point instead of a frontier
- Let R3 build the baseline their own policy is measured against
- Publish from a `-dirty` run

---

[README](../README.md) · [CONTRIBUTING](./CONTRIBUTING.md) · [ROADMAP](./ROADMAP.md) · [ROLES](./ROLES.md)
