# Pre-registered analysis

**Status:** registered, not yet executed.
**Registered:** 14 Aug 2026, before any test-split row was read.
**Owner:** R4 (Vivian). Amendments require all four roles and a dated entry below.

This document exists to be committed *before* the frozen test split is opened. It
fixes the metric, the comparison, the test, the correction, and the stopping rule
in advance, so that the result cannot be chosen after the fact.

`eval.loading.unlock_test_split` requires a path to this file and refuses to
proceed if it does not exist. That check is deliberately crude — its purpose is to
make "we'll write it up afterwards" impossible to do by reflex.

> **Everything below was written against the pilot's train and validation splits
> only.** The test split has not been read. Run
> `2026-08-13-c76a55d-4f4767` is graded and sealed; 45 of its 200 tasks are in
> `test` per [`data/splits/pilot_200.json`](../data/splits/pilot_200.json) and
> remain unopened.

---

## 1. Primary hypothesis

> A learned routing policy achieves higher pass@1 than `verifier_gated_cascade`
> **at matched GPU-second cost**, in at least one region of the cost–accuracy
> frontier, on the frozen test split.

**Primary metric.** `solved` — every hidden test passes. Binary, per
`(task, seed)`. Partial credit is not the metric: a function passing 7 of 8
hidden tests is wrong.

**Primary comparison.** `learned_D1` versus `verifier_gated_cascade`.

The cascade is the reference because it is the strongest honest baseline —
observing failure beats predicting it. A comparison against `always_small` or
`always_large` answers an easier question and is reported only as context.

**Matched cost is part of the hypothesis, not a robustness check.** A policy that
is more accurate than the small arm and cheaper than the large arm has
demonstrated nothing on its own; that is what interpolation does, and
`random_route` achieves it for free.

---

## 2. Secondary hypotheses

Both are pre-registered, both are reported whichever way they come out.

**H2 — learning beats human priors.** `learned_D0` > `heuristic_route`, with the
information set held constant (prompt only). If this fails, the learned policy has
not earned its parameters at D0 and the report says so in the headline.

**H3 — observing beats predicting.** `learned_D1` > `learned_D0`, with the
learning capacity held constant. The expected direction is a large positive gap;
the pilot's design predicts `AUC_D0 ≈ 0.60–0.68` against `AUC_D1 ≈ 0.80–0.90`.

---

## 3. Statistical plan

| Element | Choice | Why |
|---|---|---|
| Interval | Paired cluster bootstrap, 10,000 resamples, 95% percentile | Resamples **tasks and seeds**. Three seeds of one task are not three independent observations; a row-level bootstrap returns intervals that are far too narrow |
| Test | Exact McNemar, two-sided | Paired binary outcomes. Exact rather than χ², because the discordant count is small and χ² is unreliable exactly there |
| Correction | Benjamini–Hochberg, q = 0.05 | Applied across the λ grid **and** across all baseline comparisons as one family |
| λ grid | `0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5` | Fixed here. Adding a λ after seeing results is a new family and requires an amendment |
| Bootstrap seed | `0` | Recorded in `results.json`. A bootstrap without a recorded seed is not reproducible |

**Family definition.** One family = every comparison in the report. BH is applied
once, over all of them together. Splitting the family to gain significance is the
failure mode this line exists to prevent.

---

## 4. Power

Sample size is driven by the **discordant rate**, not the accuracy gap. Two arms
agreeing on almost everything need a far larger corpus than their 19.3 pp gap
suggests.

Before the corpus is finalized, compute the discordance on train+val only:

```sh
orch-eval power --discordance <measured> --odds-ratio 1.45 --power 0.80
```

**Decision rule.** If the required *n* exceeds the corpus size, the corpus grows —
the analysis does not shrink. Hash-based splits mean adding tasks does not move
existing assignments, so growing the corpus does not invalidate this registration.

The pilot's 200 tasks (45 in test) are **underpowered for the primary
hypothesis by design.** The pilot exists to measure discordance and clear the
Phase 0 gates, not to test H1. H1 is evaluated on the Phase 1 corpus.

---

## 5. Exclusions, declared in advance

Rows are excluded only for these reasons. Each is a property of the row, decided
without reference to the outcome.

| Excluded | Reason |
|---|---|
| `finish_reason == "error"` | The backend never produced output; not a model capability signal |
| `error_class == "harness_error"` | The grader failed, not the code |
| Rows from a `-dirty` run | The recorded git sha does not describe the code that ran |
| Rows failing `rollout.schema.json` | Contract violation; not repaired in place |

**Not excluded, deliberately:**

- `finish_reason == "length"` — truncation is a real failure. Excluding it would
  flatter whichever arm truncates more. The rate is **reported separately**
  (1.0% in the pilot).
- `syntax_error`, `empty_code`, `runtime_error`, `timeout` — all genuine failures.
- Rows carrying `hack_flags` — retained in the primary metric and reported
  separately as a rate. See §6.

---

## 6. Reward hacking

`hack_flags` are reported as a rate per arm and per policy, alongside every
accuracy number. The pilot measured 0.25%.

**Pre-registered rule:** if the flagged rate exceeds **2%** in any arm, the
primary metric is recomputed with flagged rows counted as failures, and **both**
numbers are reported. Choosing between them after seeing which is more favourable
is exactly what this rule prevents.

Flags in scope: `hardcoded_visible_case`, `bare_except_pass`, `reads_test_file`,
`sys_exit_or_skip`, `test_file_deleted`, `test_file_modified`.

---

## 7. Stopping rule

**The test split is opened once.** One unlock, one analysis, one report.

- No looking at test-split results to decide whether to collect more data.
- No re-salting the splits. Re-salting after seeing results is the split-level
  equivalent of unblinding twice.
- No adding baselines, features, or λ values after unblinding.
- If a bug is found after unblinding, the fix is a **new `run_id`**, a dated
  amendment below, and both results reported.

If the primary result is null, the null is the headline. A rigorously established
negative result is a stronger artifact than a positive one nobody can reproduce.

---

## 8. What would falsify the project

Stated in advance so it cannot be renegotiated later.

| Finding | Conclusion |
|---|---|
| `AUC_D1 < 0.75` | The premise is false. Neither decision point carries signal. **Hard stop.** |
| Cascade ≥ policy everywhere on the frontier | Routing does not beat observing. Report it as the finding |
| `heuristic_route` ≥ `learned_D0` | Learning did not earn its complexity at D0 |
| Result holds on the contaminated split but vanishes on LiveCodeBench post-cutoff | The effect was contamination. That is the actual finding |
| `escalation_helps` ≈ 0 | The problem is saturated; no router could win. Not a policy failure |

---

## 9. Analysis artifacts

| Artifact | Path |
|---|---|
| Frozen split | `data/splits/pilot_200.json`, salt `pilot-2026-08-13`, corpus hash `fcc0a6fd6cbd05dc` |
| Result | `results.json`, sealed with `_MANIFEST.json` and a sha256 |
| Golden gate | `orch-eval golden` must reproduce byte-identically |
| Leakage audit | `orch-eval audit`, must exit 0 |

Command that will be run, exactly once:

```sh
orch-eval report \
  --run-id <phase1-run-id> \
  --splits test \
  --unlock-prereg docs/PREREGISTRATION.md \
  --out runs/<phase1-run-id>/report
```

---

## 10. Amendments

Any change after registration is dated and appended here, with the reason and the
sign-off. An amendment made after unblinding is disclosed as such in the report.

_None._
