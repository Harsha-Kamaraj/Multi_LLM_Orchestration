"""Grader interface.

A grader turns (task, model output) into a `Grade` by *executing* something —
running tests, building an image, validating a schema. No grader in this
package calls a model. That is the point: the correctness signal the policy
optimizes against must not itself be a model's opinion.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..types import Grade, Task

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


class Grader(Protocol):
    def grade(self, task: Task, output: str) -> Grade: ...


def extract_code(output: str) -> str:
    """Pull Python source out of a model response.

    Prefers the longest fenced block (models often emit a short usage example
    alongside the real implementation). Falls back to the raw text when the
    model answered with bare code and no fence.
    """
    blocks = _FENCE.findall(output)
    if blocks:
        return max(blocks, key=len).strip()
    return output.strip()
