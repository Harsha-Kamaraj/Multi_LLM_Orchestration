"""Backend contract conformance, and the mock's planted signal."""

from __future__ import annotations

import pytest

from orchestrator.workers.backends import GenRequest, available, get_backend
from orchestrator.workers.backends.base import Backend, RawGeneration
from orchestrator.workers.backends.mock import MockBackend
from orchestrator.workers.errors import BackendError, BackendUnavailable
from orchestrator.workers.params import GREEDY


def reqs(n=4, role="small", arm="direct_small", seed=0):
    return [
        GenRequest(task_id=f"t{i}", arm=arm, seed=seed, model_role=role,
                   system="sys", user=f"solve {i}", params=GREEDY)
        for i in range(n)
    ]


def test_every_backend_is_registered():
    assert set(available()) == {"mock", "vllm_offline", "vllm_openai", "anthropic"}


def test_unknown_backend_raises():
    with pytest.raises(BackendUnavailable, match="unknown backend"):
        get_backend("nope")


@pytest.mark.parametrize("module,cls", [
    ("vllm_offline", "VLLMOfflineBackend"),
    ("vllm_openai", "VLLMOpenAIBackend"),
    ("anthropic_api", "AnthropicBackend"),
])
def test_backend_modules_import_without_touching_hardware(module, cls):
    """Importing this package must stay free on a laptop with no GPU: vLLM
    allocates VRAM at construction, so construction has to be lazy."""
    mod = __import__(f"orchestrator.workers.backends.{module}", fromlist=[cls])
    assert issubclass(getattr(mod, cls), Backend)


def test_results_are_positionally_aligned():
    """A backend that batches internally must not mismatch responses to tasks."""
    backend = get_backend("mock")
    requests = reqs(6)
    results = backend.generate(requests)
    assert len(results) == len(requests)


def test_a_misaligned_backend_is_caught():
    class Broken(MockBackend):
        def _generate(self, requests, batch_size):
            return list(super()._generate(requests, batch_size))[:-1]

    with pytest.raises(BackendError, match="positionally aligned"):
        Broken().generate(reqs(3))


def test_a_wholesale_failure_becomes_rows_not_an_exception():
    """A sweep that dies on a transient backend error loses hours of GPU time
    and resumes at the same cell."""
    class Exploding(MockBackend):
        def _generate(self, requests, batch_size):
            raise RuntimeError("engine died")

    results = Exploding().generate(reqs(3))
    assert len(results) == 3
    assert all(r.finish_reason == "error" for r in results)
    assert all("engine died" in (r.error or "") for r in results)


def test_finish_reasons_are_normalized_by_the_base_class():
    class Weird(MockBackend):
        def _generate(self, requests, batch_size):
            return [RawGeneration(text="x", model_id="m", finish_reason="end_turn")
                    for _ in requests]

    assert Weird().generate(reqs(1))[0].finish_reason == "stop"


def test_mode_and_batch_size_are_stamped_by_the_base_class():
    """A backend must not be able to label batched output as serving output
    and make its wall-clock look like latency."""
    class Liar(MockBackend):
        def _generate(self, requests, batch_size):
            return [RawGeneration(text="x", model_id="m", mode="serving",
                                  batch_size=1) for _ in requests]

    result = Liar().generate(reqs(5))[0]
    assert result.mode == "sweep" and result.batch_size == 5


def test_roles_resolve_to_model_ids():
    backend = get_backend("mock", small_model="tiny", large_model="huge")
    assert backend.model_id("small") == "tiny"
    assert backend.model_id("large") == "huge"
    with pytest.raises(ValueError, match="unknown model role"):
        backend.model_id("medium")


def test_empty_request_list_is_a_no_op():
    assert get_backend("mock").generate([]) == []


# -- the mock's guarantees ---------------------------------------------------


def test_mock_is_deterministic_regardless_of_batch_composition():
    """Results must depend only on cell identity — not on how many cells were
    generated before, and not on batch shape."""
    backend = get_backend("mock")
    batched = backend.generate(reqs(8))[3]
    alone = backend.generate([reqs(8)[3]])[0]
    assert batched.text == alone.text
    assert batched.prefill_tokens == alone.prefill_tokens


def test_mock_plants_a_skill_gap_clearing_the_phase_0_gate():
    """`A_large - A_small >= 8pp` is Phase 0's gate. The mock sits clear of it
    so a pipeline test exercises a passing path."""
    backend = get_backend("mock")
    solved = {}
    for arm, role in (("direct_small", "small"), ("direct_large", "large")):
        results = backend.generate([
            GenRequest(task_id=f"task{i}", arm=arm, seed=s, model_role=role,
                       system="s", user="u", params=GREEDY)
            for i in range(300) for s in (0, 1, 2)
        ])
        solved[arm] = sum("simulated correct" in r.text for r in results) / len(results)
    gap = solved["direct_large"] - solved["direct_small"]
    assert gap >= 0.08, f"planted gap is only {gap:.1%}"


def test_mock_produces_within_task_seed_variance():
    """Without it, R4's cluster bootstrap has nothing to cluster over."""
    backend = get_backend("mock")
    disagreements = 0
    for i in range(200):
        texts = {
            backend.generate([GenRequest(
                task_id=f"task{i}", arm="direct_small", seed=s,
                model_role="small", system="s", user="u", params=GREEDY)])[0].text
            for s in (0, 1, 2)
        }
        disagreements += len(texts) > 1
    assert disagreements > 0


def test_mock_exercises_truncation_refusal_and_error_paths():
    results = get_backend("mock").generate([
        GenRequest(task_id=f"t{i}", arm="direct_small", seed=0, model_role="small",
                   system="s", user="u", params=GREEDY)
        for i in range(3000)
    ])
    reasons = {r.finish_reason for r in results}
    assert {"stop", "length", "refusal", "error"} <= reasons


def test_mock_emits_a_registered_solution_when_it_solves():
    backend = get_backend("mock")
    backend.register_solution("task0", "def add(a, b):\n    return a + b")
    texts = [
        backend.generate([GenRequest(
            task_id="task0", arm="direct_small", seed=s, model_role="small",
            system="s", user="u", params=GREEDY)])[0].text
        for s in range(6)
    ]
    assert any("return a + b" in t for t in texts)


def test_anthropic_backend_declares_it_cannot_honor_seeds():
    """At temperature 0 a seed-blind backend returns identical text for every
    seed; the sweep warns rather than letting it surface in week 4."""
    from orchestrator.workers.backends.anthropic_api import AnthropicBackend

    assert AnthropicBackend.honors_seed is False
    assert MockBackend.honors_seed is True


def test_offline_backend_refuses_a_batch_spanning_two_models():
    """Swapping weights mid-batch defeats continuous batching and risks two
    models resident at once."""
    from orchestrator.workers.backends.vllm_offline import VLLMOfflineBackend

    backend = VLLMOfflineBackend(small_model="a", large_model="b")
    mixed = [
        GenRequest(task_id="t0", arm="direct_small", seed=0, model_role="small",
                   system="s", user="u", params=GREEDY),
        GenRequest(task_id="t1", arm="direct_large", seed=0, model_role="large",
                   system="s", user="u", params=GREEDY),
    ]
    results = backend.generate(mixed)
    assert all("spans 2 models" in (r.error or "") for r in results)
