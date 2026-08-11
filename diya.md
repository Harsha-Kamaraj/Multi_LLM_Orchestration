# R2 — Verifier & Data · Diya

You own correctness. Every claim this project makes is downstream of your grader
being right, so you are the only role whose bugs are silent.

**Your paths:** `src/orchestrator/graders/`, `data/tasks/`
**Your commit scopes:** `grader`, `data`
**You block:** R1's sweep (your task manifest) — the only true cross-role dependency in Phase 1
**You are blocked by:** nobody

---

## Mandate

Decide, executably, whether a piece of code is correct — and produce a task corpus
clean enough that "correct" means something.

You are the reason this project isn't LLM-as-judge. Protect that.

---

## You are the security boundary

You execute untrusted model output. Treat every generation as hostile, because
eventually one will be, and it won't announce itself.

```
docker run --rm --network none --memory 512m --pids-limit 128 --cpus 1
  --read-only --tmpfs /tmp:rw,size=64m -v {workdir}:/work:rw -w /work {image}
```

Every flag there is load-bearing:

| Flag | Stops |
|---|---|
| `--network none` | Exfiltration, and tests that silently pass by phoning home |
| `--read-only` + `--tmpfs` | Writes that persist between graded tasks |
| `--memory 512m` | A memory bomb taking down the host mid-sweep |
| `--pids-limit 128` | Fork bombs |
| `--cpus 1` | One task starving the sweep |

**No credentials in the grader's environment. Ever.** The subprocess backend must
pass a minimal env — `PATH`, `HOME`, `PYTHONDONTWRITEBYTECODE` — and nothing else.
A generation that reads `os.environ` should find nothing worth having.

> The subprocess backend is for local iteration only. **It must never produce
> reported numbers.** If a number reaches R4, it came from Docker.

---

## The visible/hidden split is the whole project

This is the single most important thing you own, and it's easy to get subtly wrong.

- **Visible tests** — given to the model, usable at inference by the policy and the
  cascade. This is what makes the D1 decision non-trivial: the router gets to observe
  a partial signal before deciding whether to escalate.
- **Hidden tests** — labels only. **Never a feature.** Not directly, and not
  transitively.

Enforce it **in code, not by convention.** Different fields, different accessors,
and a grader that physically cannot return hidden results into a feature path. R3
will not mean to leak them; the schema should make it impossible anyway.

---

## Interface you must satisfy

```python
grade(task, code) -> Grade {
    visible_passed, visible_total,     # usable at inference
    hidden_passed, hidden_total,       # labels only — never a feature
    error_class, hack_flags, duration_s
}
```

**You receive code, not raw model output.** R1 does the extraction. If you're
parsing markdown fences, a prompt change on R1's side becomes a bug in your file —
that dependency was removed deliberately, don't reintroduce it.

**Grading is a pure function.** Same `(task, code)` → same `Grade`, on any machine,
in any order. Any test that depends on execution order, wall-clock, network, or a
previous task's leftovers is a bug in your harness, not a flaky test.

---

## Reward-hack detection

A model that passes tests without solving the task is the most dangerous failure
mode here, because it looks exactly like success in every aggregate metric.

Flag at minimum:

- Hardcoded returns matching visible test cases
- `try/except: pass` swallowing everything
- Rewriting, monkeypatching, or deleting the test file
- `sys.exit(0)` / `pytest.skip` before assertions run
- Reading the test file at runtime

`hack_flags` ships on every row. R4 reports the rate. A rising hack rate is a
finding, not noise.

---

## Task corpus

- **EvalPlus (HumanEval+ / MBPP+)** — the `+` matters. Original HumanEval tests are
  weak enough that wrong solutions pass; EvalPlus's extra cases are the point.
- **Contamination filtering** — document what you filtered and how. R4 needs it for
  the leakage audit.
- **Split at task level**, 60/20/20, manifest **hashed and committed**.
- **LiveCodeBench post-cutoff** for Phase 3's clean replication.

You build and hash the task corpus. **R4 owns `data/splits/`** — the split is an
evaluation artifact, and the person who proves the result should control what's
frozen. Hand over the corpus; don't hold the split.

---

## Week 1 (Phase 0)

- [ ] Sign off on `schemas/` by day 3
- [ ] Task manifest for the 200-task pilot — **this is R1's blocker, ship it first**
- [ ] Docker grader working end to end, per-test pass counts
- [ ] Grade R1's pilot sweep
- [ ] A deliberately malicious solution suite, and a hack detector that catches it

---

## Definition of done

Grading is a pure function, container-isolated, and a deliberately malicious
solution suite is caught by the hack detector.

Write the malicious suite yourself, first. A detector validated only against honest
code has been validated against nothing.

---

## What you must not do

- Let a subprocess-graded number reach R4
- Return hidden test outcomes anywhere a feature builder can reach them
- Parse raw model output — you take code
- Mutate a graded row; a re-grade is a new `run_id`
- Open the frozen test split — R4 only
