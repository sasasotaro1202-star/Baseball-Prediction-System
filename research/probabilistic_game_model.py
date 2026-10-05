"""Research-only posterior-predictive game modelling primitives.

This module is the first integration layer for the hierarchical generative
architecture described in the project research source. It deliberately reuses
existing game-level simulators instead of duplicating their state-transition
logic.

Design:
    latent talent -> current state/scenario -> game simulator
        -> posterior predictive mixture -> uncertainty diagnostics

The module is NOT a production model. It must not be interpreted as evidence
that the underlying talent/state assumptions are empirically superior.
Production eligibility still requires PIT/OOS/calibration/robustness/holdout
gates in the repository governance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "probabilistic-game-model-v1"
RESEARCH_STATUS = "RESEARCH_ONLY"
SUPPORTED_TARGETS = ("NPB", "MLB")
EPS = 1e-12


@dataclass(frozen=True)
class PosteriorEstimate:
    """A one-dimensional normal-normal posterior summary.

    observed_var is the variance of an individual observation. The supplied
    observation_count converts it into the variance of an observed sample mean.
    This is a deliberately explicit shrinkage primitive; it is not a claim
    that every baseball skill follows a Gaussian likelihood.
    """

    mean: float
    variance: float
    prior_mean: float
    prior_variance: float
    observed_mean: float
    observed_var: float
    observation_count: int


@dataclass(frozen=True)
class Scenario:
    """A mutually exclusive pre-game scenario with an explicit prior weight."""

    scenario_id: str
    weight: float
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionContract:
    """Minimum audit metadata for a research posterior-predictive run."""

    game_id: str
    target: str
    prediction_time: str
    prediction_cutoff: str
    available_at: str
    pit_status: str
    model_version: str = SCHEMA_VERSION
    production_eligible: bool = False
    status: str = RESEARCH_STATUS


def _as_finite(value: float, name: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        out = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("timestamp must not be empty")
        out = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if out.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return out


def assert_pit_ready(
    *,
    pit_status: str,
    available_at: str | datetime | None,
    prediction_cutoff: str | datetime | None,
) -> None:
    """Fail closed unless availability is explicitly proven before cutoff."""

    if str(pit_status).upper() != "PASS":
        raise ValueError(f"PIT status is not PASS: {pit_status}")
    if available_at is None or prediction_cutoff is None:
        raise ValueError("PIT requires available_at and prediction_cutoff")
    available = _parse_time(available_at)
    cutoff = _parse_time(prediction_cutoff)
    if available > cutoff:
        raise ValueError(
            f"PIT violation: available_at={available.isoformat()} > "
            f"prediction_cutoff={cutoff.isoformat()}"
        )


def normal_normal_update(
    *,
    prior_mean: float,
    prior_variance: float,
    observed_mean: float,
    observed_variance: float,
    observation_count: int,
) -> PosteriorEstimate:
    """Return the conjugate posterior for a normal prior/normal observations.

    The function is useful for explicit, auditable shrinkage of latent
    components. Larger observation counts increase the data precision while
    small samples remain anchored to the prior.
    """

    pm = _as_finite(prior_mean, "prior_mean")
    pv = _as_finite(prior_variance, "prior_variance")
    om = _as_finite(observed_mean, "observed_mean")
    ov = _as_finite(observed_variance, "observed_variance")
    n = int(observation_count)

    if pv <= 0:
        raise ValueError("prior_variance must be > 0")
    if ov <= 0:
        raise ValueError("observed_variance must be > 0")
    if n <= 0:
        raise ValueError("observation_count must be > 0")

    prior_precision = 1.0 / pv
    data_precision = n / ov
    posterior_variance = 1.0 / (prior_precision + data_precision)
    posterior_mean = posterior_variance * (
        pm * prior_precision + om * data_precision
    )

    return PosteriorEstimate(
        mean=float(posterior_mean),
        variance=float(posterior_variance),
        prior_mean=pm,
        prior_variance=pv,
        observed_mean=om,
        observed_var=ov,
        observation_count=n,
    )


def apply_state_adjustment(
    talent_mean: float,
    *,
    fatigue: float = 0.0,
    fatigue_sensitivity: float = 0.0,
    state_shift: float = 0.0,
) -> float:
    """Convert a latent talent estimate into a transparent current-state mean."""

    t = _as_finite(talent_mean, "talent_mean")
    f = _as_finite(fatigue, "fatigue")
    s = _as_finite(fatigue_sensitivity, "fatigue_sensitivity")
    shift = _as_finite(state_shift, "state_shift")
    return float(t - f * s + shift)


def normalize_scenarios(scenarios: Sequence[Scenario]) -> tuple[Scenario, ...]:
    """Validate and normalize a finite scenario mixture."""

    if not scenarios:
        raise ValueError("at least one scenario is required")
    if len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("scenario_id must be unique")

    total = 0.0
    checked: list[Scenario] = []
    for scenario in scenarios:
        weight = _as_finite(scenario.weight, f"weight[{scenario.scenario_id}]")
        if weight <= 0:
            raise ValueError(f"scenario weight must be > 0: {scenario.scenario_id}")
        total += weight
        checked.append(
            Scenario(
                scenario_id=str(scenario.scenario_id),
                weight=weight,
                parameters=dict(scenario.parameters),
            )
        )

    if total <= 0 or not math.isfinite(total):
        raise ValueError("scenario weights must have a finite positive total")

    return tuple(
        Scenario(s.scenario_id, float(s.weight / total), s.parameters)
        for s in checked
    )


def mix_probability_vectors(
    probabilities: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> np.ndarray:
    """Compute an explicit weighted mixture of probability vectors."""

    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError("probabilities must be a non-empty 2-D array")
    if w.ndim != 1 or len(w) != p.shape[0]:
        raise ValueError("weights length must match probability rows")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("weights must be finite and > 0")
    row_sum = p.sum(axis=1)
    if np.any(row_sum <= 0):
        raise ValueError("each probability vector must have positive mass")
    p = p / row_sum[:, None]
    w = w / w.sum()
    out = (p * w[:, None]).sum(axis=0)
    out = np.clip(out, 0.0, 1.0)
    out /= max(float(out.sum()), EPS)
    return out


def binary_entropy(p: float) -> float:
    q = float(np.clip(p, EPS, 1.0 - EPS))
    return float(-(q * math.log(q) + (1.0 - q) * math.log(1.0 - q)))


def uncertainty_decomposition(
    *,
    scenario_probabilities: Sequence[float],
    scenario_weights: Sequence[float],
) -> dict[str, float]:
    """Decompose binary-outcome variance into within/between-scenario parts.

    This is the law of total variance:
        Var(Y) = E[Var(Y | scenario)] + Var(E[Y | scenario])

    The first term is an aleatoric/randomness component and the second is a
    scenario/epistemic component. The labels are intentionally operational,
    not ontological claims about the true data-generating process.
    """

    p = np.asarray(scenario_probabilities, dtype=float)
    w = np.asarray(scenario_weights, dtype=float)
    if p.ndim != 1 or w.ndim != 1 or len(p) != len(w) or len(p) == 0:
        raise ValueError("scenario probabilities and weights must be 1-D and aligned")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("scenario probabilities must lie in [0, 1]")
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("scenario weights must be finite and > 0")

    w = w / w.sum()
    mean_p = float(np.dot(w, p))
    within = float(np.dot(w, p * (1.0 - p)))
    between = float(np.dot(w, (p - mean_p) ** 2))
    total = float(mean_p * (1.0 - mean_p))
    reconstruction_error = abs(total - (within + between))
    return {
        "mean_probability": mean_p,
        "aleatoric_variance": within,
        "epistemic_scenario_variance": between,
        "total_binary_variance": total,
        "variance_reconstruction_error": reconstruction_error,
        "scenario_entropy": float(np.dot(w, np.array([binary_entropy(x) for x in p]))),
        "mixture_entropy": binary_entropy(mean_p),
    }


def model_disagreement_summary(
    model_probabilities: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Summarize distributional disagreement without choosing a winner."""

    p = np.asarray(model_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2:
        raise ValueError("at least two model probability vectors are required")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("model probabilities must be finite and non-negative")
    p = p / np.maximum(p.sum(axis=1, keepdims=True), EPS)
    mean = p.mean(axis=0)
    variance = p.var(axis=0)
    pairwise_js: list[float] = []

    def _entropy(v: np.ndarray) -> float:
        q = np.clip(v, EPS, 1.0)
        return float(-(q * np.log(q)).sum())

    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            m = 0.5 * (p[i] + p[j])
            pairwise_js.append(
                max(0.0, _entropy(m) - 0.5 * _entropy(p[i]) - 0.5 * _entropy(p[j]))
            )

    return {
        "model_count": int(len(p)),
        "mean_probability": mean.tolist(),
        "probability_variance": variance.tolist(),
        "max_component_sd": float(np.sqrt(variance).max()),
        "mean_pairwise_js": float(np.mean(pairwise_js)),
        "max_pairwise_js": float(max(pairwise_js)),
    }


def _allocate_scenarios(simulations: int, weights: np.ndarray) -> np.ndarray:
    """Allocate draws proportionally with at least one draw per scenario."""
    n = len(weights)
    if simulations < n:
        raise ValueError("simulations must be at least the number of scenarios")
    remaining = simulations - n
    if remaining == 0:
        return np.ones(n, dtype=int)
    raw = remaining * weights
    base = np.floor(raw).astype(int)
    remainder = raw - base
    extra = remaining - int(base.sum())
    order = np.argsort(-remainder, kind="mergesort")
    base[order[:extra]] += 1
    return base + 1


def _score_matrix(scores: np.ndarray, max_runs: int) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("simulator output must have shape (n, 2)")
    if len(scores) == 0:
        raise ValueError("simulator returned no samples")
    if not np.isfinite(scores).all():
        raise ValueError("simulator produced non-finite scores")
    rounded = np.rint(scores)
    if np.any(np.abs(scores - rounded) > 1e-9):
        raise ValueError("simulator scores must be integer-valued")
    if np.any(scores < 0):
        raise ValueError("simulator scores must be non-negative")
    if np.any(rounded > max_runs):
        raise ValueError("simulator score exceeds configured max_runs")
    matrix = np.zeros((max_runs + 1, max_runs + 1), dtype=float)
    for home, away in rounded.astype(int):
        matrix[home, away] += 1.0
    matrix /= max(float(matrix.sum()), EPS)
    return matrix


def _top4(matrix: np.ndarray) -> list[dict[str, float | str]]:
    flat = matrix.ravel()
    indices = np.argsort(-flat, kind="mergesort")[:4]
    width = matrix.shape[1]
    return [
        {
            "score": f"{int(i // width)}-{int(i % width)}",
            "probability": float(flat[i]),
        }
        for i in indices
    ]


def posterior_predictive_run(
    scenarios: Sequence[Scenario],
    *,
    simulate_fn: Callable[[Scenario, np.random.Generator, int], np.ndarray],
    simulations: int = 20_000,
    seed: int = 42,
    target: str = "NPB",
    max_runs: int = 30,
    pit_status: str = "PASS",
    available_at: str | datetime | None = None,
    prediction_cutoff: str | datetime | None = None,
    model_probabilities: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Run a posterior-predictive scenario mixture over a game simulator.

    simulate_fn must return integer [home_score, away_score] samples. The
    allocation across scenarios is sampled from the validated scenario weights.
    This preserves explicit scenario uncertainty rather than collapsing an
    unknown starter/weather/etc. to one deterministic input.
    """

    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target: {target}")
    if int(simulations) <= 0:
        raise ValueError("simulations must be > 0")
    if int(max_runs) < 7:
        raise ValueError("max_runs must be >= 7")

    assert_pit_ready(
        pit_status=pit_status,
        available_at=available_at,
        prediction_cutoff=prediction_cutoff,
    )
    normalized = normalize_scenarios(scenarios)
    if int(simulations) < len(normalized):
        raise ValueError("simulations must be at least the number of scenarios")
    rng = np.random.default_rng(int(seed))
    # Stratified deterministic allocation preserves the declared scenario
    # mixture and still leaves the inner game simulation stochastic.
    allocations = _allocate_scenarios(
        int(simulations), np.array([s.weight for s in normalized], dtype=float)
    )

    all_scores: list[np.ndarray] = []
    scenario_reports: list[dict[str, Any]] = []
    for scenario, count in zip(normalized, allocations):
        child_seed = int(rng.integers(0, 2**63 - 1))
        child = np.random.default_rng(child_seed)
        samples = np.asarray(simulate_fn(scenario, child, int(count)), dtype=float)
        if samples.ndim != 2 or samples.shape != (int(count), 2):
            raise ValueError(
                f"scenario {scenario.scenario_id} returned "
                f"{samples.shape}, expected {(int(count), 2)}"
            )
        all_scores.append(samples)
        home_p = float(np.mean(samples[:, 0] > samples[:, 1]))
        draw_p = float(np.mean(samples[:, 0] == samples[:, 1]))
        away_p = float(np.mean(samples[:, 0] < samples[:, 1]))
        scenario_reports.append(
            {
                "scenario_id": scenario.scenario_id,
                "weight": float(scenario.weight),
                "allocated_simulations": int(count),
                "outcome_probability": {
                    "home": home_p,
                    "draw": draw_p,
                    "away": away_p,
                },
            }
        )

    if not all_scores:
        raise RuntimeError("scenario allocation produced no simulator samples")

    scores = np.vstack(all_scores)
    matrix = _score_matrix(scores, int(max_runs))

    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    low = float(
        sum(
            matrix[h, a]
            for h in range(matrix.shape[0])
            for a in range(matrix.shape[1])
            if h + a <= 6
        )
    )

    if target == "MLB" and draw > 0:
        raise ValueError(
            f"MLB simulator produced tied terminal scores ({draw:.6f}); "
            "binary MLB target requires an explicit extra-inning rule"
        )

    class_names = ("home", "draw", "away") if target == "NPB" else ("home", "away")
    class_probs = (
        np.array([home, draw, away], dtype=float)
        if target == "NPB"
        else np.array([home, away], dtype=float)
    )
    if class_probs.sum() <= 0:
        raise RuntimeError("invalid posterior-predictive class mass")
    class_probs /= class_probs.sum()

    mc_home_se = math.sqrt(max(EPS, home * (1.0 - home)) / len(scores))
    uncertainty = uncertainty_decomposition(
        scenario_probabilities=[r["outcome_probability"]["home"] for r in scenario_reports],
        scenario_weights=[r["weight"] for r in scenario_reports],
    )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": RESEARCH_STATUS,
        "production_eligible": False,
        "target": target,
        "simulations_requested": int(simulations),
        "simulations_observed": int(len(scores)),
        "seed": int(seed),
        "scenario_count": int(len(normalized)),
        "scenario_reports": scenario_reports,
        "probabilities": {
            "home_win": home,
            "away_win": away,
            **({"draw": draw} if target == "NPB" else {}),
            "low_le_6": low,
            "high_ge_7": float(1.0 - low),
        },
        "class_names": list(class_names),
        "class_probabilities": class_probs.tolist(),
        "score_distribution": matrix.tolist(),
        "top4_exact_score": _top4(matrix),
        "mc": {
            "home_win_se": float(mc_home_se),
            "sample_count": int(len(scores)),
            "monte_carlo_precision_note": "simulation error is distinct from model/data uncertainty",
        },
        "uncertainty": {
            **uncertainty,
            "unallocated_scenario_probability": 0.0,
        },
    }
    if model_probabilities is not None:
        result["model_disagreement"] = model_disagreement_summary(model_probabilities)
    return result


def build_prediction_contract(
    *,
    game_id: str,
    target: str,
    prediction_time: str,
    prediction_cutoff: str,
    available_at: str,
    pit_status: str,
    model_version: str = SCHEMA_VERSION,
) -> PredictionContract:
    """Build and validate the research-only prediction metadata contract."""

    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target: {target}")
    assert_pit_ready(
        pit_status=pit_status,
        available_at=available_at,
        prediction_cutoff=prediction_cutoff,
    )
    # Prediction time is validated separately because available_at and cutoff
    # alone do not establish that the prediction itself happened after cutoff.
    prediction = _parse_time(prediction_time)
    cutoff = _parse_time(prediction_cutoff)
    if prediction < cutoff:
        raise ValueError(
            f"prediction_time={prediction.isoformat()} precedes cutoff={cutoff.isoformat()}"
        )
    return PredictionContract(
        game_id=str(game_id),
        target=target,
        prediction_time=prediction.isoformat(),
        prediction_cutoff=cutoff.isoformat(),
        available_at=_parse_time(available_at).isoformat(),
        pit_status="PASS",
        model_version=str(model_version),
    )


__all__ = [
    "SCHEMA_VERSION",
    "RESEARCH_STATUS",
    "PredictionContract",
    "PosteriorEstimate",
    "Scenario",
    "apply_state_adjustment",
    "assert_pit_ready",
    "build_prediction_contract",
    "mix_probability_vectors",
    "model_disagreement_summary",
    "normal_normal_update",
    "normalize_scenarios",
    "posterior_predictive_run",
    "uncertainty_decomposition",
]
