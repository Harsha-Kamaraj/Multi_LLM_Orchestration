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
never be silently averaged together downstream, and `validate_imputation`
refuses to regress against anything that is not `serving`.

Imports here are kept light on purpose: `vllm`, `transformers`, `anthropic`,
and `pyarrow` are all resolved lazily, so importing this package costs nothing
on a machine with no GPU and no model weights.
"""

from __future__ import annotations

__all__ = [
    # errors
    "WorkerError", "BackendError", "BackendUnavailable", "ParamsDriftError",
    "StoreError", "UnsealedRunError", "CharacterizationError",
    # the record and its vocabulary
    "Generation", "SCHEMA_VERSION", "normalize_finish_reason",
    # configuration
    "GenParams", "GREEDY", "SAMPLED",
    "Arm", "ARMS", "FROZEN_LADDER", "DEFAULT_SEEDS", "get_arm", "resolve_arms",
    "PromptContext", "PromptTemplate", "TEMPLATES", "get_template",
    # generation
    "Backend", "GenRequest", "RawGeneration", "get_backend", "available",
    "Extraction", "extract", "extract_code",
    # run identity and storage
    "make_run_id", "is_publishable", "is_valid",
    "RolloutStore", "read_generations", "read_manifest", "read_rows", "list_runs",
    "ResumeIndex", "build_resume_index",
    # the sweep
    "SweepConfig", "SweepReport", "run_sweep",
    "CorpusView", "build_corpus", "load_tasks", "load_splits",
    # cost
    "CostCoefficients", "LinearFit", "Observation", "fit_linear",
    "ImputationReport", "validate_imputation",
    "characterize", "CharacterizationResult",
    "impute_run", "ImputeReport",
]

from .arms import (
    ARMS, DEFAULT_SEEDS, FROZEN_LADDER, Arm, get_arm, resolve_arms,
)
from .backends import Backend, GenRequest, RawGeneration, available, get_backend
from .characterize import CharacterizationResult, characterize
from .corpus import CorpusView, build_corpus, load_splits, load_tasks
from .cost import (
    CostCoefficients, ImputationReport, LinearFit, Observation, fit_linear,
    validate_imputation,
)
from .errors import (
    BackendError, BackendUnavailable, CharacterizationError, ParamsDriftError,
    StoreError, UnsealedRunError, WorkerError,
)
from .extract import Extraction, extract, extract_code
from .generation import SCHEMA_VERSION, Generation, normalize_finish_reason
from .impute import ImputeReport, impute_run
from .params import GREEDY, SAMPLED, GenParams
from .prompts import TEMPLATES, PromptContext, PromptTemplate, get_template
from .resume import ResumeIndex, build_resume_index
from .runid import is_publishable, is_valid, make_run_id
from .store import (
    RolloutStore, list_runs, read_generations, read_manifest, read_rows,
)
from .sweep import SweepConfig, SweepReport, run_sweep
