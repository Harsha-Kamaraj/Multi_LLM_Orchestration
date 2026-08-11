"""Resume must be exact in both directions.

Recomputing a finished cell wastes GPU hours. Accepting a cell computed under
different parameters merges two experiments into one store, and nothing
downstream can tell.
"""

from __future__ import annotations

import pytest

from orchestrator.workers.errors import ParamsDriftError
from orchestrator.workers.generation import Generation
from orchestrator.workers.resume import build_resume_index, pending, summarize
from orchestrator.workers.store import RolloutStore

RUN_ID = "2026-08-11-abc1234-def456"
PH = "c0ff9e5c88e4"


def gen(task_id, arm="direct_small", seed=0, params_hash=PH, finish="stop"):
    return Generation(
        run_id=RUN_ID, task_id=task_id, arm=arm, seed=seed,
        params_hash=params_hash, code="x = 1", model_id="mock-small",
        finish_reason=finish,
    )


def _write(runs_root, gens):
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append_many(gens)
    store.close()


def test_missing_run_yields_an_empty_index(runs_root):
    """First invocation and a resume take the same code path, so the
    resuming path is exercised on every sweep."""
    assert len(build_resume_index(runs_root, RUN_ID)) == 0


def test_completed_cells_are_recognized(runs_root):
    _write(runs_root, [gen(f"t{i}") for i in range(4)])
    index = build_resume_index(runs_root, RUN_ID)
    assert index.has(("t2", "direct_small", 0, PH))
    assert len(index) == 4


def test_a_different_params_hash_is_not_a_hit(runs_root):
    """The whole point of the key: a template edit must not reuse old cells."""
    _write(runs_root, [gen("t0")])
    index = build_resume_index(runs_root, RUN_ID)
    assert not index.has(("t0", "direct_small", 0, "deadbeef0000"))


def test_a_different_seed_is_not_a_hit(runs_root):
    _write(runs_root, [gen("t0", seed=0)])
    assert not build_resume_index(runs_root, RUN_ID).has(("t0", "direct_small", 1, PH))


def test_a_different_arm_is_not_a_hit(runs_root):
    _write(runs_root, [gen("t0", arm="direct_small")])
    assert not build_resume_index(runs_root, RUN_ID).has(("t0", "direct_large", 0, PH))


def test_errors_are_retried(runs_root):
    """A backend timeout is worth another attempt."""
    _write(runs_root, [gen("t0", finish="error")])
    index = build_resume_index(runs_root, RUN_ID)
    assert not index.has(("t0", "direct_small", 0, PH))
    assert ("t0", "direct_small", 0, PH) in index.retryable


@pytest.mark.parametrize("finish", ["length", "refusal"])
def test_real_outcomes_are_not_retried(runs_root, finish):
    """Retrying a refusal until it changes selects for the lucky sample and
    biases the arm's measured accuracy upward."""
    _write(runs_root, [gen("t0", finish=finish)])
    assert build_resume_index(runs_root, RUN_ID).has(("t0", "direct_small", 0, PH))


def test_retry_errors_can_be_disabled(runs_root):
    _write(runs_root, [gen("t0", finish="error")])
    index = build_resume_index(runs_root, RUN_ID, retry_errors=False)
    assert index.has(("t0", "direct_small", 0, PH))


def test_a_later_success_supersedes_an_earlier_error(runs_root):
    _write(runs_root, [gen("t0", finish="error"), gen("t0", finish="stop")])
    index = build_resume_index(runs_root, RUN_ID)
    assert index.has(("t0", "direct_small", 0, PH))
    assert ("t0", "direct_small", 0, PH) not in index.retryable


def test_params_drift_is_refused(runs_root):
    _write(runs_root, [gen("t0", params_hash="aaaaaaaaaaaa")])
    index = build_resume_index(runs_root, RUN_ID)
    with pytest.raises(ParamsDriftError, match="params_hash"):
        index.check_params({"direct_small": PH})


def test_matching_params_pass_the_check(runs_root):
    _write(runs_root, [gen("t0")])
    build_resume_index(runs_root, RUN_ID).check_params({"direct_small": PH})


def test_two_params_hashes_for_one_arm_is_refused(runs_root):
    _write(runs_root, [gen("t0"), gen("t1", params_hash="bbbbbbbbbbbb")])
    with pytest.raises(ParamsDriftError):
        build_resume_index(runs_root, RUN_ID).check_params({"direct_small": PH})


def test_pending_preserves_order(runs_root):
    """Deterministic order means an interrupted sweep resumes where it stopped
    instead of scattering across the corpus."""
    _write(runs_root, [gen("t1")])
    index = build_resume_index(runs_root, RUN_ID)
    cells = [(f"t{i}", "direct_small", 0, PH) for i in range(4)]
    assert pending(cells, index) == [
        ("t0", "direct_small", 0, PH),
        ("t2", "direct_small", 0, PH),
        ("t3", "direct_small", 0, PH),
    ]


def test_summary_reports_remaining_work(runs_root):
    _write(runs_root, [gen(f"t{i}") for i in range(3)])
    summary = summarize(build_resume_index(runs_root, RUN_ID), planned=10)
    assert summary["cells_complete"] == 3 and summary["cells_remaining"] == 7


def test_rows_without_identity_are_counted_not_dropped_silently(runs_root):
    """A non-zero count here means something wrote rows this package did not."""
    store = RolloutStore(runs_root, RUN_ID).open()
    store.append(gen("t0"))
    store.close()
    with store.part_files()[0].open("a", encoding="utf-8", newline="") as fh:
        fh.write('{"arm": "direct_small", "seed": 0}\n')
    assert build_resume_index(runs_root, RUN_ID).n_torn == 1
