"""The cost model — characterize once, impute everywhere.

Three tiers, in descending order of authority:

1. **Tokens.** Primary. Hardware-independent and reproducible on anyone's
   machine. Already on every row.
2. **GPU-seconds.** Secondary. `prefill_tokens x a + decode_tokens x b`, with
   `a` and `b` fitted per model.
3. **USD.** Tertiary. GPU-seconds times an instance rate, always stated as an
   assumption rather than presented as a measurement.

The coefficients come from a **separate characterization pass** on the
OpenAI-compatible server at declared concurrency — never from a sweep. This is
what lets the numbers survive new hardware: re-characterize, and every existing
rollout's imputed cost updates without regenerating a single token.

**GPU-seconds excludes the intercept; imputed latency includes it.** They are
different quantities. The intercept is per-request overhead — scheduling, HTTP,
detokenization — which a caller waits for but which does not scale with the
work done. Folding it into GPU-seconds would charge every short generation for
a fixed cost the GPU did not spend, and would inflate the apparent cost of
exactly the cheap arm the routing question is about.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import CharacterizationError
from .generation import Generation

COEFFICIENTS_VERSION = 1

# Default filename, under `bench/`. Committed to the repo: it is a measured
# artifact the whole project depends on, and it belongs in version control
# next to the code that produced it.
DEFAULT_COEFFICIENTS_PATH = Path("bench") / "cost_coefficients.json"

# Concurrency whose fit defines "GPU-seconds" and imputed latency. Batch 1 is
# the only setting where elapsed time is uncontended, so it is the only honest
# basis for a per-request cost.
REFERENCE_BATCH = 1

# Below this, a three-parameter fit is describing noise. The pass raises rather
# than emitting coefficients that carry an R-squared and no information.
MIN_FIT_SAMPLES = 12


@dataclass(frozen=True)
class LinearFit:
    """`seconds = intercept + prefill x a + decode x b`, plus how well it holds."""

    intercept_s: float
    prefill_s_per_token: float
    decode_s_per_token: float
    n: int
    r2: float
    rmse_s: float
    #: R-squared on samples excluded from the fit. The honest number: an
    #: in-sample R-squared can be high because the model memorized noise.
    holdout_r2: float | None = None
    holdout_n: int = 0

    def predict(self, prefill_tokens: int, decode_tokens: int) -> float:
        return (
            self.intercept_s
            + prefill_tokens * self.prefill_s_per_token
            + decode_tokens * self.decode_s_per_token
        )

    def occupancy(self, prefill_tokens: int, decode_tokens: int) -> float:
        """Work-proportional GPU time, with the fixed per-request cost removed."""
        return max(0.0, (
            prefill_tokens * self.prefill_s_per_token
            + decode_tokens * self.decode_s_per_token
        ))


@dataclass
class CostCoefficients:
    """Fitted coefficients for every characterized model.

    `usd_per_gpu_hour` is an assumption, recorded alongside the measurements so
    a reader can substitute their own rate without re-deriving anything.
    """

    models: dict[str, dict[int, LinearFit]] = field(default_factory=dict)
    hardware: str = "unspecified"
    usd_per_gpu_hour: float = 0.0
    reference_batch: int = REFERENCE_BATCH
    version: int = COEFFICIENTS_VERSION
    created_at: str = ""
    notes: str = ""

    # -- lookup --------------------------------------------------------------

    def fit_for(self, model_id: str, batch: int | None = None) -> LinearFit:
        """Coefficients for a model at a batch size, defaulting to the reference.

        A missing model raises. Falling back to another model's coefficients
        would produce cost numbers that look fine and describe different
        weights — the exact failure this whole module exists to prevent.
        """
        by_batch = self.models.get(model_id)
        if not by_batch:
            raise KeyError(
                f"no cost coefficients for model {model_id!r}; characterized "
                f"models are {sorted(self.models)}. Run the characterization "
                f"pass on this model before imputing its cost."
            )
        want = self.reference_batch if batch is None else batch
        if want in by_batch:
            return by_batch[want]
        if self.reference_batch in by_batch:
            return by_batch[self.reference_batch]
        return by_batch[min(by_batch)]

    def has(self, model_id: str) -> bool:
        return bool(self.models.get(model_id))

    # -- imputation ----------------------------------------------------------

    def gpu_seconds(self, model_id: str, prefill_tokens: int,
                    decode_tokens: int) -> float:
        return self.fit_for(model_id).occupancy(prefill_tokens, decode_tokens)

    def imputed_latency_s(self, model_id: str, prefill_tokens: int,
                          decode_tokens: int) -> float:
        return self.fit_for(model_id).predict(prefill_tokens, decode_tokens)

    def usd(self, model_id: str, prefill_tokens: int, decode_tokens: int) -> float:
        """Dollars, under the recorded instance rate. Zero when no rate is set."""
        gpu_s = self.gpu_seconds(model_id, prefill_tokens, decode_tokens)
        return gpu_s * (self.usd_per_gpu_hour / 3600.0)

    def impute(self, gen: Generation) -> Generation:
        """Attach `gpu_seconds` and `imputed_latency_s` to a generation.

        Returns the row unchanged when its model has no coefficients, rather
        than raising. A sweep must be able to run before characterization
        exists — the fields stay null and a later pass fills them in from the
        stored token counts, which is the whole point of imputing rather than
        measuring.
        """
        if not self.has(gen.model_id):
            return gen
        return gen.with_cost(
            gpu_seconds=self.gpu_seconds(
                gen.model_id, gen.prefill_tokens, gen.decode_tokens),
            imputed_latency_s=self.imputed_latency_s(
                gen.model_id, gen.prefill_tokens, gen.decode_tokens),
        )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "hardware": self.hardware,
            "usd_per_gpu_hour": self.usd_per_gpu_hour,
            "reference_batch": self.reference_batch,
            "notes": self.notes,
            "models": {
                model: {str(batch): asdict(fit) for batch, fit in sorted(by_batch.items())}
                for model, by_batch in sorted(self.models.items())
            },
        }

    def save(self, path: Path | str = DEFAULT_COEFFICIENTS_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="",
        )
        return path

    @staticmethod
    def load(path: Path | str = DEFAULT_COEFFICIENTS_PATH) -> "CostCoefficients":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no cost coefficients at {path}; run the characterization pass "
                f"first — imputed cost is not available until a model has been "
                f"measured on the serving stack"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        models: dict[str, dict[int, LinearFit]] = {}
        for model, by_batch in (data.get("models") or {}).items():
            models[model] = {
                int(batch): LinearFit(**fit) for batch, fit in by_batch.items()
            }
        return CostCoefficients(
            models=models,
            hardware=data.get("hardware", "unspecified"),
            usd_per_gpu_hour=float(data.get("usd_per_gpu_hour") or 0.0),
            reference_batch=int(data.get("reference_batch") or REFERENCE_BATCH),
            version=int(data.get("version") or COEFFICIENTS_VERSION),
            created_at=data.get("created_at", ""),
            notes=data.get("notes", ""),
        )

    @staticmethod
    def load_or_empty(path: Path | str = DEFAULT_COEFFICIENTS_PATH) -> "CostCoefficients":
        """Load if present, otherwise an empty set that imputes nothing."""
        try:
            return CostCoefficients.load(path)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            return CostCoefficients()


# -- fitting -----------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One measured serving request: tokens in, tokens out, seconds elapsed."""

    prefill_tokens: int
    decode_tokens: int
    seconds: float


def fit_linear(observations: Sequence[Observation], *,
               holdout_fraction: float = 0.25) -> LinearFit:
    """Least-squares fit of seconds against prefill and decode tokens.

    A deterministic every-fourth-sample holdout, not a random one: the fit must
    be reproducible from the same observations, and a random split would make
    the reported R-squared differ between two runs over identical data.

    The holdout is the number that matters. Fitting three parameters to a few
    dozen points can produce a high in-sample R-squared from noise alone; only
    the held-out figure says whether the imputation predicts requests it has
    not seen.
    """
    import numpy as np

    n = len(observations)
    if n < MIN_FIT_SAMPLES:
        raise CharacterizationError(
            f"{n} observations is too few to fit three parameters "
            f"(minimum {MIN_FIT_SAMPLES}); collect a wider probe set rather "
            f"than reporting coefficients fitted to noise"
        )

    holdout_idx: set[int] = set()
    if holdout_fraction > 0 and n >= MIN_FIT_SAMPLES * 2:
        step = max(2, int(round(1.0 / holdout_fraction)))
        holdout_idx = set(range(step - 1, n, step))

    train = [o for i, o in enumerate(observations) if i not in holdout_idx]
    holdout = [o for i, o in enumerate(observations) if i in holdout_idx]

    def design(obs: Sequence[Observation]):
        return (
            np.array([[1.0, o.prefill_tokens, o.decode_tokens] for o in obs], dtype=float),
            np.array([o.seconds for o in obs], dtype=float),
        )

    X, y = design(train)
    # `rcond=None` uses the machine-precision cutoff, so a rank-deficient
    # design is reported through `rank` instead of being silently regularized.
    beta, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    if rank < 3:
        raise CharacterizationError(
            "the probe set does not vary prefill and decode independently "
            f"(design rank {rank} < 3), so the two coefficients are not "
            "separately identifiable. Vary prompt length and max_tokens "
            "independently across probes."
        )

    intercept, a, b = (float(v) for v in beta)
    r2, rmse = _r2_rmse(X @ beta, y)

    holdout_r2: float | None = None
    if holdout:
        Xh, yh = design(holdout)
        holdout_r2 = _r2_rmse(Xh @ beta, yh)[0]

    return LinearFit(
        intercept_s=intercept,
        prefill_s_per_token=a,
        decode_s_per_token=b,
        n=len(train),
        r2=r2,
        rmse_s=rmse,
        holdout_r2=holdout_r2,
        holdout_n=len(holdout),
    )


def _r2_rmse(predicted: Any, actual: Any) -> tuple[float, float]:
    """Coefficient of determination and RMSE.

    Returns `r2 = 0.0` when the actual values have no variance: with nothing to
    explain, a perfect fit is not evidence of anything, and reporting 1.0 would
    overstate the model.
    """
    import numpy as np

    resid = np.asarray(actual) - np.asarray(predicted)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((np.asarray(actual) - np.mean(actual)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / len(resid))) if len(resid) else 0.0
    return r2, rmse


# -- the definition-of-done check --------------------------------------------


@dataclass(frozen=True)
class ImputationReport:
    """Whether imputed latency actually tracks measured wall-clock.

    R1's definition of done is `R^2 > 0.9` here. It is the honesty check on the
    whole cost model: if it is low, the imputation is fiction, and that has to
    be said out loud before R4 builds a frontier on top of it.
    """

    model_id: str
    n: int
    r2: float
    rmse_s: float
    mean_measured_s: float
    mean_imputed_s: float
    passed: bool
    threshold: float

    @property
    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] {self.model_id}: R^2={self.r2:.4f} "
            f"(threshold {self.threshold:.2f}), n={self.n}, "
            f"RMSE={self.rmse_s:.3f}s, "
            f"measured={self.mean_measured_s:.3f}s vs imputed={self.mean_imputed_s:.3f}s"
        )


def validate_imputation(generations: Iterable[Generation],
                        coefficients: CostCoefficients,
                        *, threshold: float = 0.9,
                        batch: int | None = None) -> list[ImputationReport]:
    """Compare imputed latency against measured wall-clock, per model.

    **Only `mode == "serving"` rows are eligible.** A batched sweep's
    wall-clock is a queue-depth measurement; regressing imputed latency against
    it would produce a number that is not about latency at all. Rows with
    approximate token counts are excluded for the same reason — the regressor
    would be measuring the tokenizer's error.
    """
    import numpy as np

    eligible: dict[str, list[Generation]] = {}
    for gen in generations:
        if gen.mode != "serving" or not gen.tokens_exact:
            continue
        if batch is not None and gen.batch_size != batch:
            continue
        if gen.wall_ms <= 0 or not coefficients.has(gen.model_id):
            continue
        eligible.setdefault(gen.model_id, []).append(gen)

    reports = []
    for model_id, gens in sorted(eligible.items()):
        measured = np.array([g.wall_ms / 1000.0 for g in gens], dtype=float)
        imputed = np.array([
            coefficients.imputed_latency_s(
                model_id, g.prefill_tokens, g.decode_tokens)
            for g in gens
        ], dtype=float)
        r2, rmse = _r2_rmse(imputed, measured)
        reports.append(ImputationReport(
            model_id=model_id,
            n=len(gens),
            r2=r2,
            rmse_s=rmse,
            mean_measured_s=float(np.mean(measured)),
            mean_imputed_s=float(np.mean(imputed)),
            passed=r2 > threshold,
            threshold=threshold,
        ))
    return reports
