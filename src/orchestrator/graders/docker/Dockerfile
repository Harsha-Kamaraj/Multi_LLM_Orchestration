# The grading sandbox image. pytest is baked in at build time deliberately —
# `--network none` (the whole point of the sandbox, see diya.md) means a
# container can't `pip install` anything at run time. An image that expects
# to fetch its own test runner over the network it's forbidden from using is
# a grader that silently stops grading and reports every task as a harness
# error, which looks exactly like "the sandbox is broken" instead of what it
# actually is: an image built wrong.
#
# Build:  docker build -t orch-grader:py312 -f src/orchestrator/graders/docker/Dockerfile .
# Rebuild whenever the pinned versions below change; the grader's default
# `image=` argument names this tag.

FROM python:3.12-slim

RUN pip install --no-cache-dir --no-compile pytest==8.3.* pytest-timeout==2.3.*

WORKDIR /work
