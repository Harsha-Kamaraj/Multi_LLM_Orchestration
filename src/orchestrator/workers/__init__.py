"""R1 — serving and workers.

Turns `(task_id, arm, seed)` into a `Generation` record, at throughput,
reproducibly, and produces the cost coefficients that let the rest of the
project talk about money without owning a GPU.

Two modes live here and are deliberately kept apart:

* **Sweep** — offline batch generation tuned for throughput. Wall-clock is
  recorded but is *not* a latency measurement, because a batched request's
  wall-clock is a function of queue depth rather than of the model.
* **Serving / characterization** — a request-response server at a declared,
  fixed concurrency. This is the only mode whose wall-clock means anything.

Every `Generation` carries `mode` and `batch_size` precisely so the two can
never be silently averaged together downstream.
"""

from __future__ import annotations

__all__ = ["WorkerError"]

from .errors import WorkerError
