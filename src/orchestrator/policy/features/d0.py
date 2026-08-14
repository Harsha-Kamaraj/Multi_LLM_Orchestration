"""D0 — everything knowable before a single token is generated.

Expect these to be weak. `AUC_D0` lands around 0.60–0.68, and that is the
structure of the problem rather than a modelling failure: predicting whether a
model will solve a task, from the task alone, is genuinely hard. The asymmetry
against D1 is the project's central finding, so a D0 set tuned until it looks
respectable would destroy the thing being measured.

These features also carry a specific burden. `learned_D0` sits one rung above
`heuristic_route` on the capacity ladder with the *same information* and far
more free parameters, so this set has to justify dozens of parameters against
two thresholds. Adding features until D0 wins would be answering the wrong
question — if the tuned heuristic ties it, that is the finding.

The signals here are deliberately the obvious ones: length, test count, and
surface structure of the prompt. That is not a lack of ambition. It is the
honest comparison — an interviewer's first question is "couldn't you have just
used prompt length?", and the only way to answer it is to have used prompt
length.

## Everything here comes from the join, not the row

The rollout row carries no prompt. These features read `task_prompt`,
`task_entrypoint`, and `task_visible_tests`, which `load_rollouts` joins from
R2's task manifest. Building D0 features without `tasks_path` fails loudly in
`FeatureSet.build` rather than producing a matrix of zeros.
"""

from __future__ import annotations

import re
from typing import Mapping

from .spec import FeatureSet, feature

#: Words that suggest a task needs an algorithm rather than a transformation.
#: Chosen before looking at any outcome, and deliberately short: a keyword list
#: grown by checking which words improve AUC is a model fitted by hand, on the
#: labels, without a validation split.
_ALGORITHMIC = (
    "graph", "tree", "dynamic", "recursive", "recursion", "optimize",
    "optimal", "minimum", "maximum", "shortest", "permutation", "combination",
    "matrix", "prime", "sort", "search",
)

_LOOP_KEYWORDS = ("for", "while", "each", "every", "iterate", "repeat")

_WORD = re.compile(r"\w+")
_SENTENCE = re.compile(r"[.!?]+")


def _text(view: Mapping[str, object], key: str) -> str:
    value = view[key]
    return value if isinstance(value, str) else ("" if value is None else str(value))


# -- size --------------------------------------------------------------------


@feature("prompt_chars", "D0", "task_prompt",
         description="Prompt length in characters.")
def prompt_chars(view: Mapping[str, object]) -> float:
    return float(len(_text(view, "task_prompt")))


@feature("prompt_words", "D0", "task_prompt",
         description="Prompt length in words — the interviewer's first guess.")
def prompt_words(view: Mapping[str, object]) -> float:
    return float(len(_WORD.findall(_text(view, "task_prompt"))))


@feature("prompt_sentences", "D0", "task_prompt",
         description="Sentence count; a proxy for how many constraints there are.")
def prompt_sentences(view: Mapping[str, object]) -> float:
    text = _text(view, "task_prompt").strip()
    if not text:
        return 0.0
    return float(len([s for s in _SENTENCE.split(text) if s.strip()]))


@feature("prompt_mean_word_len", "D0", "task_prompt",
         description="Mean word length, standing in for vocabulary difficulty.")
def prompt_mean_word_len(view: Mapping[str, object]) -> float:
    words = _WORD.findall(_text(view, "task_prompt"))
    return float(sum(len(w) for w in words) / len(words)) if words else 0.0


# -- the visible suite -------------------------------------------------------


@feature("n_visible_tests", "D0", "task_visible_tests",
         description="How many visible tests the model is shown.")
def n_visible_tests(view: Mapping[str, object]) -> float:
    text = _text(view, "task_visible_tests")
    return float(len([line for line in text.splitlines() if line.strip()]))


@feature("visible_test_chars", "D0", "task_visible_tests",
         description="Size of the visible suite, not just its count.")
def visible_test_chars(view: Mapping[str, object]) -> float:
    return float(len(_text(view, "task_visible_tests")))


@feature("visible_assert_density", "D0", "task_visible_tests",
         description="Asserts per visible test line.")
def visible_assert_density(view: Mapping[str, object]) -> float:
    text = _text(view, "task_visible_tests")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return float(text.count("assert") / len(lines))


# -- surface structure -------------------------------------------------------


@feature("algorithmic_keywords", "D0", "task_prompt",
         description="Count of algorithm-suggesting words in the prompt.")
def algorithmic_keywords(view: Mapping[str, object]) -> float:
    lowered = _text(view, "task_prompt").lower()
    return float(sum(lowered.count(word) for word in _ALGORITHMIC))


@feature("loop_keywords", "D0", "task_prompt",
         description="Words implying iteration, which correlate with control flow.")
def loop_keywords(view: Mapping[str, object]) -> float:
    words = [w.lower() for w in _WORD.findall(_text(view, "task_prompt"))]
    return float(sum(1 for w in words if w in _LOOP_KEYWORDS))


@feature("has_code_block", "D0", "task_prompt",
         description="Whether the prompt contains a fenced example.")
def has_code_block(view: Mapping[str, object]) -> float:
    return float("```" in _text(view, "task_prompt"))


@feature("digit_density", "D0", "task_prompt",
         description="Fraction of characters that are digits; flags numeric specs.")
def digit_density(view: Mapping[str, object]) -> float:
    text = _text(view, "task_prompt")
    if not text:
        return 0.0
    return float(sum(c.isdigit() for c in text) / len(text))


@feature("entrypoint_len", "D0", "task_entrypoint",
         description="Length of the required function name.")
def entrypoint_len(view: Mapping[str, object]) -> float:
    return float(len(_text(view, "task_entrypoint")))


@feature("entrypoint_words", "D0", "task_entrypoint",
         description="Underscore-separated words in the entrypoint name.")
def entrypoint_words(view: Mapping[str, object]) -> float:
    name = _text(view, "task_entrypoint")
    return float(len([p for p in name.split("_") if p])) if name else 0.0


#: The default D0 set.
#:
#: `task_x_d0` is deliberately **not** here. It exists only on the synthetic
#: fixture, and a feature set that behaves differently on fixture and real data
#: is one that was never really tested. Tests that want the planted proxy
#: directly build a one-feature set for it.
D0_FEATURES = FeatureSet(
    [
        prompt_chars,
        prompt_words,
        prompt_sentences,
        prompt_mean_word_len,
        n_visible_tests,
        visible_test_chars,
        visible_assert_density,
        algorithmic_keywords,
        loop_keywords,
        has_code_block,
        digit_density,
        entrypoint_len,
        entrypoint_words,
    ],
    name="d0_prompt",
)
