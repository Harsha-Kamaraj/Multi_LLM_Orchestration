"""`params_hash` is row identity. These tests are the contract on it."""

from __future__ import annotations

import pytest

from orchestrator.workers.params import GREEDY, SAMPLED, GenParams
from orchestrator.workers.prompts import TEMPLATES


def test_hash_is_stable_across_construction():
    assert GenParams().params_hash == GenParams().params_hash
    assert GenParams(temperature=0.0).params_hash == GREEDY.params_hash


def test_hash_is_stable_across_processes():
    """Pinned literal, so a change to the recipe cannot pass unnoticed.

    If this fails, every previously-written row's `params_hash` has become
    unreachable and resume will regenerate an entire corpus. That is a
    deliberate, reviewed change — never an incidental one.
    """
    assert GREEDY.params_hash == "c0ff9e5c88e4"


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.7),
    ("top_p", 0.9),
    ("max_tokens", 4096),
    ("top_k", 40),
    ("min_p", 0.05),
    ("stop", ("###",)),
    ("seeded", False),
    ("template_id", "direct_notests_v1"),
])
def test_every_generation_affecting_field_changes_the_hash(field, value):
    """A setting that changes output but not the hash would let a resume
    reuse cells generated under different conditions."""
    assert GREEDY.evolve(**{field: value}).params_hash != GREEDY.params_hash


def test_template_text_is_hashed_not_its_label(monkeypatch):
    """Editing a template without renaming it must change the hash.

    This is the mechanism that stops a mid-sweep prompt tweak from silently
    poisoning a store.
    """
    before = GREEDY.params_hash
    edited = TEMPLATES["direct_v1"].__class__(
        template_id="direct_v1",
        system=TEMPLATES["direct_v1"].system,
        user=TEMPLATES["direct_v1"].user + "\n\nBe concise.",
    )
    monkeypatch.setitem(TEMPLATES, "direct_v1", edited)
    assert GREEDY.params_hash != before


def test_renaming_a_template_without_editing_it_keeps_the_hash(monkeypatch):
    """The converse: identical text generates identically and should reuse."""
    original = TEMPLATES["direct_v1"]
    renamed = original.__class__(
        template_id="direct_v1_renamed",
        system=original.system,
        user=original.user,
    )
    monkeypatch.setitem(TEMPLATES, "direct_v1", renamed)
    assert GREEDY.params_hash == "c0ff9e5c88e4"


def test_model_id_is_not_in_the_hash():
    """The arm already carries the model, and the arm is in the resume key."""
    assert "model" not in GREEDY.hash_payload()


def test_hash_payload_is_the_documented_set():
    """A field silently entering or leaving the hash should show in a diff."""
    assert set(GREEDY.hash_payload()) == {
        "v", "template_hash", "temperature", "top_p", "max_tokens",
        "top_k", "min_p", "stop", "seeded",
    }


def test_greedy_and_sampled_are_distinguishable():
    assert GREEDY.params_hash != SAMPLED.params_hash


def test_evolve_does_not_mutate():
    before = GREEDY.params_hash
    GREEDY.evolve(max_tokens=99)
    assert GREEDY.params_hash == before
