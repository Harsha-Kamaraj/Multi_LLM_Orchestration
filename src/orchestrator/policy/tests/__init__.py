"""R3's tests.

They live inside the package rather than under a top-level `tests/` because
the pre-commit hook derives a commit's scope from its staged paths: anything
under `src/orchestrator/policy/` is scope `policy`, and a root `tests/`
directory would resolve to `infra`. A test and the code it covers belong to the
same owner and therefore to the same commit. R1 solved the same problem by
putting theirs under `bench/`.
"""
