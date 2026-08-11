"""Turn a model response into Python source.

Extraction is R1's, not R2's. If it lived in the grader, changing a prompt
template would break R2's tests and R1 would own a bug in someone else's file.
Keeping fence-parsing on this side is what makes prompt format entirely R1's
business — R2 receives code and grades what it is given.

The hard case is not "find the fence". It is **choosing among several fences**.
Models routinely emit the implementation and then a short usage example, or a
first attempt and then a correction. Picking the longest block, which is the
obvious heuristic, picks wrong whenever the example is chatty or the correction
is terse. So candidates are scored on whether they parse, whether they define
the task's entrypoint, and only then on length.

Every extraction reports the strategy that produced it. Extraction quality is
measurable rather than assumed: a run whose `bare_*` rate moves is a prompt
regression, and it shows up in the manifest instead of looking like a model
capability gap.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# Opening fence, optional language tag, body, then either a closing fence or
# end of string. The trailing-`\Z` alternative is what catches the common
# truncation case: `finish_reason == "length"` cuts the response mid-block and
# the closing fence never arrives.
_FENCE = re.compile(
    r"^[ \t]*```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:^[ \t]*```|\Z)",
    re.DOTALL | re.MULTILINE,
)

# Tags that are definitely not the solution. `text` and `output` are how models
# label expected-output blocks; running one as Python is guaranteed nonsense.
_NON_PYTHON_TAGS = frozenset({
    "bash", "sh", "shell", "console", "text", "output", "json", "yaml", "yml",
    "toml", "ini", "diff", "sql", "html", "css", "js", "javascript", "ts",
    "typescript", "c", "cpp", "java", "go", "rust", "makefile", "dockerfile",
})

_PYTHON_TAGS = frozenset({"python", "py", "python3", "pycon", ""})


@dataclass(frozen=True)
class Extraction:
    """Extracted source plus how it was recovered.

    `strategy` and `parses` are logged on every row. They separate "the model
    wrote bad code" from "we failed to find the code the model wrote", which
    otherwise look identical in a pass rate.
    """

    code: str
    strategy: str
    n_blocks: int = 0
    parses: bool = False
    defines_entrypoint: bool = False

    @property
    def ok(self) -> bool:
        """Whether this is worth sending to the grader at all."""
        return bool(self.code.strip())


def _normalize(text: str) -> str:
    """Line endings and BOM only — never content.

    CRLF matters: a stray `\r` inside a source line is legal in Python but
    makes hashes differ between a sweep run on Windows and one on Linux, for
    generations that are otherwise identical.
    """
    return text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def _try_dedent(code: str) -> str:
    """Undo a uniform indent, but only when that is what is wrong.

    Models sometimes emit a block indented as if it were nested in prose. The
    dedent is attempted only if the original fails to parse and the dedented
    version succeeds, so a correctly-indented block is never touched.
    """
    lines = [ln for ln in code.split("\n") if ln.strip()]
    if not lines:
        return code
    indents = []
    for ln in lines:
        stripped = ln.lstrip(" \t")
        indents.append(ln[: len(ln) - len(stripped)])
    common = indents[0]
    for ind in indents[1:]:
        while not ind.startswith(common):
            common = common[:-1]
            if not common:
                return code
    if not common:
        return code
    return "\n".join(
        ln[len(common):] if ln.startswith(common) else ln
        for ln in code.split("\n")
    )


def _defines(code: str, name: str) -> bool:
    """Whether `name` is bound at module level by a def, class, or assignment."""
    if not name:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        # Fall back to a textual check so a block that nearly parses is still
        # recognised as the one carrying the entrypoint.
        return re.search(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b",
                         code, re.MULTILINE) is not None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return True
    return False


def _has_definition(code: str) -> bool:
    """Whether the block defines anything at all, versus being a call example."""
    return re.search(r"^\s*(?:async\s+)?(?:def|class)\s+\w", code, re.MULTILINE) is not None


def _is_imports_only(code: str) -> bool:
    """Whether a block is nothing but import statements.

    Models sometimes emit imports in their own block and the implementation in
    the next one. Such a block is safe to prepend to the winner because it
    cannot redefine anything — which is what makes this fix narrower, and much
    safer, than concatenating blocks in general.
    """
    stripped = code.strip()
    if not stripped:
        return False
    try:
        tree = ast.parse(stripped)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return bool(tree.body) and all(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body
    )


def _blocks(text: str) -> list[tuple[str, str]]:
    """All fenced blocks as `(tag, body)`, in document order."""
    return [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(text)]


def _score(code: str, tag: str, index: int, entrypoint: str) -> tuple:
    """Rank a candidate block. Higher sorts better.

    Ordered by decreasing authority, so a shorter block that defines the
    entrypoint always beats a longer one that does not:

    1. defines the entrypoint
    2. parses as Python
    3. defines anything at all
    4. explicitly tagged `python` rather than untagged
    5. longer
    6. later in the response — a correction follows what it corrects
    """
    return (
        _defines(code, entrypoint),
        _parses(code),
        _has_definition(code),
        tag != "",
        len(code),
        index,
    )


def extract(output: str, entrypoint: str = "") -> Extraction:
    """Recover Python source from a model response.

    `entrypoint` is the symbol the task's tests import. Supplying it is what
    makes the "implementation plus usage example" case resolve correctly, so
    pass it whenever the task has one.
    """
    text = _normalize(output)
    if not text.strip():
        return Extraction(code="", strategy="empty", n_blocks=0)

    blocks = _blocks(text)
    candidates = [
        (tag, body) for tag, body in blocks
        if tag not in _NON_PYTHON_TAGS
    ]

    if candidates:
        scored = []
        for i, (tag, body) in enumerate(candidates):
            code = body.strip("\n").rstrip()
            if not _parses(code):
                dedented = _try_dedent(code)
                if _parses(dedented):
                    code = dedented
            scored.append((_score(code, tag, i, entrypoint), code))
        scored.sort(key=lambda t: t[0])
        best = scored[-1][1]
        strategy = "fenced"

        # An import-only block belongs to whichever block follows it. Prepend
        # any that the winner does not already carry, so a solution split as
        # "imports here, implementation there" does not lose its imports and
        # fail at import time in the sandbox.
        orphan_imports = [
            code for _s, code in scored[:-1]
            if _is_imports_only(code) and code.strip() not in best
        ]
        if orphan_imports:
            merged = "\n".join(orphan_imports) + "\n\n" + best
            if _parses(merged):
                best = merged
                strategy = "fenced_imports_merged"

        defines_ep = bool(entrypoint) and _defines(best, entrypoint)
        parses = _parses(best)

        # The entrypoint may still be missing when a model splits a solution
        # across blocks in some other way. Joining every candidate is a last
        # resort — it can pull in a usage example — so it is attempted only
        # when the alternative is handing the grader code with no entrypoint.
        if entrypoint and not defines_ep and len(scored) > 1:
            joined = "\n\n".join(code for _s, code in scored)
            if _parses(joined) and _defines(joined, entrypoint):
                return Extraction(
                    code=joined, strategy="fenced_concat", n_blocks=len(blocks),
                    parses=True, defines_entrypoint=True,
                )

        if strategy == "fenced":
            if defines_ep:
                strategy = "fenced_entrypoint"
            elif parses:
                strategy = "fenced_parsed"
            else:
                strategy = "fenced_unparsed"

        # An odd number of fences means the closing one never arrived, i.e. the
        # response was cut off. Worth its own label because it correlates with
        # `finish_reason == "length"`: it is a truncation signal, not a model
        # error, and the two must not be pooled.
        if text.count("```") % 2 == 1:
            strategy = "fenced_truncated"

        return Extraction(
            code=best, strategy=strategy, n_blocks=len(blocks),
            parses=parses, defines_entrypoint=defines_ep,
        )

    # No usable fence. The model answered with bare code, or with prose that
    # happens to contain code. Strip leading prose a line at a time until what
    # remains parses — bounded, because an unbounded search would happily
    # "recover" the last two lines of an essay.
    bare = text.strip()
    if _parses(bare):
        return Extraction(
            code=bare, strategy="bare_parsed", n_blocks=0, parses=True,
            defines_entrypoint=_defines(bare, entrypoint),
        )

    lines = bare.split("\n")
    for start in range(1, min(len(lines), 12)):
        candidate = "\n".join(lines[start:]).strip()
        if candidate and _parses(candidate) and _has_definition(candidate):
            return Extraction(
                code=candidate, strategy="bare_trimmed", n_blocks=0, parses=True,
                defines_entrypoint=_defines(candidate, entrypoint),
            )

    # Nothing parsed. Hand back the raw text anyway: the grader will record a
    # syntax error, which is the honest outcome, and suppressing the row here
    # would hide a real generation failure.
    return Extraction(
        code=bare, strategy="bare_unparsed", n_blocks=0, parses=False,
        defines_entrypoint=False,
    )


def extract_code(output: str, entrypoint: str = "") -> str:
    """Convenience wrapper for callers that only want the source."""
    return extract(output, entrypoint).code
