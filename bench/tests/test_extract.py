"""Code extraction is R1's, so its failure modes are R1's too.

The hard case is choosing among several fenced blocks, not finding one. These
cases are the real shapes models emit.
"""

from __future__ import annotations

import pytest

from orchestrator.workers.extract import extract, extract_code

ENTRY = "add"


def test_single_fenced_block():
    result = extract("```python\ndef add(a, b):\n    return a + b\n```", ENTRY)
    assert result.strategy == "fenced_entrypoint"
    assert result.parses and result.defines_entrypoint
    assert result.code == "def add(a, b):\n    return a + b"


def test_implementation_beats_a_longer_usage_example():
    """The obvious heuristic — longest block — picks wrong here."""
    output = (
        "```python\ndef add(a, b):\n    return a + b\n```\n\n"
        "Example usage, which is deliberately longer:\n\n"
        "```python\nresult = add(1, 2)\nprint(result)\nprint('and some more')\n"
        "print('padding to be the longest block')\n```"
    )
    assert extract(output, ENTRY).code == "def add(a, b):\n    return a + b"


def test_imports_in_a_separate_block_are_kept():
    """Dropping the import would fail at import time in the sandbox, which
    grades as a model error rather than as the extraction bug it is."""
    output = (
        "```python\nimport math\n```\n\nand then\n\n"
        "```python\ndef add(a, b):\n    return math.floor(a + b)\n```"
    )
    result = extract(output, ENTRY)
    assert result.strategy == "fenced_imports_merged"
    assert "import math" in result.code and result.parses


def test_non_python_blocks_are_ignored():
    output = (
        "```bash\npip install numpy\n```\n\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n\n"
        "```output\n3\n```"
    )
    assert extract(output, ENTRY).code == "def add(a, b):\n    return a + b"


def test_a_later_correction_wins_over_an_earlier_attempt():
    output = (
        "```python\ndef add(a, b):\n    return a - b\n```\n"
        "Sorry, that is wrong:\n"
        "```python\ndef add(a, b):\n    return a + b\n```"
    )
    assert "a + b" in extract(output, ENTRY).code


def test_truncation_is_labelled_separately():
    """`fenced_truncated` correlates with finish_reason == "length" and is a
    truncation signal, not a model error. Pooling them hides a max_tokens bug."""
    result = extract("Sure:\n\n```python\ndef add(a, b):\n    return a +", ENTRY)
    assert result.strategy == "fenced_truncated"
    assert not result.parses


def test_uniformly_indented_block_is_dedented():
    result = extract("```python\n    def add(a, b):\n        return a + b\n```", ENTRY)
    assert result.parses and result.defines_entrypoint


def test_bare_code_without_a_fence():
    result = extract("def add(a, b):\n    return a + b\n", ENTRY)
    assert result.strategy == "bare_parsed" and result.parses


def test_prose_before_bare_code_is_trimmed():
    result = extract("Sure! Here it is.\n\ndef add(a, b):\n    return a + b\n", ENTRY)
    assert result.strategy == "bare_trimmed" and result.parses


def test_empty_output():
    result = extract("")
    assert result.strategy == "empty" and not result.ok


def test_refusal_is_returned_unparsed_not_swallowed():
    """Returning nothing would hide a real generation failure; the grader
    recording a syntax error is the honest outcome."""
    result = extract("I can't help with that.", ENTRY)
    assert result.strategy == "bare_unparsed" and not result.parses
    assert result.code


def test_crlf_is_normalized():
    """A stray \\r changes a hash between a Windows sweep and a Linux one for
    generations that are otherwise identical."""
    assert "\r" not in extract("```python\r\ndef add(a, b):\r\n    return a\r\n```").code


def test_untagged_fence_is_treated_as_python():
    assert extract("```\ndef add(a, b):\n    return a + b\n```", ENTRY).parses


@pytest.mark.parametrize("output", [
    "", "   \n\n  ", "```python\n```", "no code at all",
    "```python", "``````", "```py\n\n```",
])
def test_never_raises_on_degenerate_input(output):
    """A sweep must not die on one strange response."""
    assert isinstance(extract(output, ENTRY).code, str)


def test_extract_code_wrapper():
    assert extract_code("```python\nx = 1\n```") == "x = 1"


def test_extraction_is_pure():
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    assert extract(output, ENTRY) == extract(output, ENTRY)
