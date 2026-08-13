"""D1 — what generating the candidate revealed.

Observing failure beats predicting it. That single sentence is why `AUC_D1` is
expected around 0.80–0.90 while `AUC_D0` sits near 0.65, and why the escalation
decision is the one worth making. The visible-test outcome is doing most of the
work here, and it should be.

Everything in `D0_FEATURES` is available at D1 too, and `D1_FEATURES` includes
it. Holding the learning fixed and varying only the information is the whole
point of the `learned_D0 → learned_D1` comparison; that comparison is only
clean if D1 is a strict superset.

## The visible outcome is a feature, the hidden outcome is the label

Both are produced by the same grader in the same pass, which is what makes this
worth stating twice. `visible_passed / visible_total` is observable at
inference — the router runs the visible tests and looks. `hidden_passed` is the
answer sheet, is what these features are fitted to predict, and has already
been removed from the rows by the loader.

## What is not here, and why

**`wall_ms`.** Rejected at declaration by `NEVER_A_FEATURE`. Under
`mode == "sweep"` it measures queue depth.

**`gpu_seconds` and `imputed_latency_s`.** Observable at D1 and deliberately
unused as *inputs* to `P_pass`. They are what `E_cost` and `E_latency` predict,
and a pass-probability head reading them would couple the two heads that the
three-head design exists to keep separate.

**Anything from ladder step k+1.** There is no repair ladder yet. When there
is, a feature reading a repair outcome at the step that decides whether to
repair is the same mistake as reading the hidden tests, and it will look like a
much better result.
"""

from __future__ import annotations

import ast
from typing import Mapping, Sequence

from .d0 import D0_FEATURES
from .spec import FeatureSet, feature

#: Finish reasons worth an indicator of their own. `length` is the important
#: one: a truncated generation is a failed generation dressed as a successful
#: one, and it grades as a capability gap.
_FINISH_REASONS = ("stop", "length", "refusal", "error")

#: Extraction strategies R1 records. A run whose `bare_*` rate moves is a
#: prompt regression rather than a model capability change, so the strategy is
#: genuinely informative about whether the candidate is trustworthy.
_BARE_PREFIX = "bare"


def _text(view: Mapping[str, object], key: str) -> str:
    value = view[key]
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _num(view: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = view[key]
    return default if value is None else float(value)


# -- the visible-test outcome ------------------------------------------------


@feature("visible_pass_rate", "D1", "visible_passed", "visible_total",
         description="Fraction of visible tests passed — the strongest D1 signal.")
def visible_pass_rate(view: Mapping[str, object]) -> float:
    total = _num(view, "visible_total")
    if total <= 0:
        # No visible suite means no observation, which is different from
        # observing zero. Both map to 0.0 here and `has_visible_tests`
        # distinguishes them, so the model can tell "failed" from "unknown".
        return 0.0
    return _num(view, "visible_passed") / total


@feature("visible_all_passed", "D1", "visible_passed", "visible_total",
         description="Whether every visible test passed. What the cascade gates on.")
def visible_all_passed(view: Mapping[str, object]) -> float:
    total = _num(view, "visible_total")
    return float(total > 0 and _num(view, "visible_passed") == total)


@feature("visible_none_passed", "D1", "visible_passed", "visible_total",
         description="Whether the candidate failed every visible test.")
def visible_none_passed(view: Mapping[str, object]) -> float:
    total = _num(view, "visible_total")
    return float(total > 0 and _num(view, "visible_passed") == 0)


@feature("has_visible_tests", "D1", "visible_total",
         description="Whether any visible test existed to observe.")
def has_visible_tests(view: Mapping[str, object]) -> float:
    return float(_num(view, "visible_total") > 0)


@feature("visible_failed_count", "D1", "visible_passed", "visible_total",
         description="Absolute number of failing visible tests.")
def visible_failed_count(view: Mapping[str, object]) -> float:
    total = _num(view, "visible_total")
    if total <= 0:
        return 0.0
    return max(0.0, total - _num(view, "visible_passed"))


# -- the candidate itself ----------------------------------------------------


@feature("code_chars", "D1", "code", description="Length of the extracted code.")
def code_chars(view: Mapping[str, object]) -> float:
    return float(len(_text(view, "code")))


@feature("code_lines", "D1", "code", description="Non-blank lines of code.")
def code_lines(view: Mapping[str, object]) -> float:
    return float(len([l for l in _text(view, "code").splitlines() if l.strip()]))


@feature("code_parses", "D1", "code_parses",
         description="Whether the extracted code is syntactically valid Python.")
def code_parses(view: Mapping[str, object]) -> float:
    return float(bool(view["code_parses"]))


@feature("code_ast_nodes", "D1", "code",
         description="AST node count — structural size, robust to formatting.")
def code_ast_nodes(view: Mapping[str, object]) -> float:
    source = _text(view, "code")
    if not source.strip():
        return 0.0
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        # Unparseable is real information, and `code_parses` carries it. Zero
        # here means "no structure to measure", not "small".
        return 0.0
    return float(sum(1 for _ in ast.walk(tree)))


@feature("code_max_depth", "D1", "code",
         description="Deepest nesting in the candidate; a complexity proxy.")
def code_max_depth(view: Mapping[str, object]) -> float:
    source = _text(view, "code")
    if not source.strip():
        return 0.0
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return 0.0

    def depth(node: ast.AST, current: int = 0) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return current
        return max(depth(child, current + 1) for child in children)

    try:
        return float(depth(tree))
    except RecursionError:
        return 0.0


@feature("defines_entrypoint", "D1", "code", "task_entrypoint",
         description="Whether the candidate defines the function the task asked for.")
def defines_entrypoint(view: Mapping[str, object]) -> float:
    entrypoint = _text(view, "task_entrypoint").strip()
    source = _text(view, "code")
    if not entrypoint or not source.strip():
        return 0.0
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        # Fall back to a textual check rather than returning 0: unparseable
        # code that clearly defines the entrypoint is a different failure from
        # code that never defines it, and the escalation decision differs.
        return float(f"def {entrypoint}" in source)
    return float(any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entrypoint
        for node in ast.walk(tree)
    ))


@feature("code_has_todo", "D1", "code",
         description="Placeholder markers a model leaves when it gives up.")
def code_has_todo(view: Mapping[str, object]) -> float:
    lowered = _text(view, "code").lower()
    return float(any(marker in lowered for marker in
                     ("todo", "fixme", "not implemented", "notimplemented",
                      "pass  #", "...")))


# -- how the generation ended ------------------------------------------------


def _finish_reason_feature(reason: str):
    @feature(f"finish_{reason}", "D1", "finish_reason",
             description=f"Whether the generation ended with finish_reason={reason}.")
    def fn(view: Mapping[str, object], _reason: str = reason) -> float:
        return float(_text(view, "finish_reason") == _reason)

    return fn


FINISH_FEATURES = [_finish_reason_feature(r) for r in _FINISH_REASONS]


@feature("decode_tokens", "D1", "decode_tokens",
         description="How much the model wrote. Correlates with both effort and waffle.")
def decode_tokens(view: Mapping[str, object]) -> float:
    return _num(view, "decode_tokens")


@feature("prefill_tokens", "D1", "prefill_tokens",
         description="Prompt length as the serving tokenizer actually counted it.")
def prefill_tokens(view: Mapping[str, object]) -> float:
    return _num(view, "prefill_tokens")


@feature("extraction_was_bare", "D1", "extract_strategy",
         description="Code recovered without a fence — a weaker extraction.")
def extraction_was_bare(view: Mapping[str, object]) -> float:
    return float(_text(view, "extract_strategy").startswith(_BARE_PREFIX))


@feature("had_error", "D1", "error",
         description="Whether the backend recorded an error for this generation.")
def had_error(view: Mapping[str, object]) -> float:
    return float(bool(_text(view, "error").strip()))


@feature("hack_flag_count", "D1", "hack_flags",
         description="Reward-hack detections. A rising rate is a finding, not noise.")
def hack_flag_count(view: Mapping[str, object]) -> float:
    flags = view["hack_flags"]
    return float(len(flags)) if flags else 0.0


# -- siblings, which are not free --------------------------------------------


@feature("sibling_visible_agreement", "D1", "visible_passed", "visible_total",
         description="How consistently sibling draws passed the visible tests.",
         needs_siblings=True, paid_arms=("probe_small",))
def sibling_visible_agreement(view: Mapping[str, object],
                              siblings: Sequence[Mapping[str, object]]) -> float:
    """Self-consistency across the other draws of this task and arm.

    The row itself is excluded by `FeatureSet.build`, which compares by
    identity. Including it would fold the outcome being predicted into the
    prediction, and the resulting AUC would look excellent.

    Returns 0.5 with no siblings — maximum uncertainty, which is honest. A
    default of 0 or 1 would be a confident claim derived from no evidence.
    """
    if not siblings:
        return 0.5
    rates = []
    for sibling in siblings:
        total = sibling["visible_total"]
        if total:
            rates.append(float(sibling["visible_passed"] or 0) / float(total))
    return float(sum(rates) / len(rates)) if rates else 0.5


@feature("sibling_count", "D1", "visible_total",
         description="How many sibling draws were available to this decision.",
         needs_siblings=True, paid_arms=("probe_small",))
def sibling_count(view: Mapping[str, object],
                  siblings: Sequence[Mapping[str, object]]) -> float:
    return float(len(siblings))


#: The default D1 set: D0 plus everything generating revealed.
#:
#: A strict superset of D0 on purpose. `learned_D0 → learned_D1` is supposed to
#: isolate *information* with learning held constant, and it cannot do that if
#: the two sets also differ in which prompt features they carry.
D1_FEATURES = FeatureSet(
    [
        # D0, re-declared at D1 so the decision-point check passes on the same
        # objects rather than on copies that could drift.
        *(
            feature(f.name, "D1", *f.source_columns, description=f.description)(f.fn)
            for f in D0_FEATURES
        ),
        visible_pass_rate,
        visible_all_passed,
        visible_none_passed,
        has_visible_tests,
        visible_failed_count,
        code_chars,
        code_lines,
        code_parses,
        code_ast_nodes,
        code_max_depth,
        defines_entrypoint,
        code_has_todo,
        *FINISH_FEATURES,
        decode_tokens,
        prefill_tokens,
        extraction_was_bare,
        had_error,
        hack_flag_count,
    ],
    name="d1_observed",
)

#: Off by default. Every feature here obliges the policy to pay for the extra
#: draws it reads, so switching it on is a cost decision as much as a modelling
#: one — see `features/__init__.feature_set`.
PROBE_FEATURES = FeatureSet(
    [sibling_visible_agreement, sibling_count],
    name="d1_probe",
)
