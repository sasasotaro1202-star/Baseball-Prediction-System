"""Research-only individually calibrated constrained ensemble challenger.

The fitter consumes strictly out-of-sample Development probabilities. It never
accepts holdout labels and the frozen spec can be applied to later probabilities
without re-fitting. This is deliberately small, deterministic, and separate
from the production Champion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import log_loss

from evaluation.calibration import TemperatureCalibration, fit_temperature


@dataclass(frozen=True)
class CalibratedBlendSpec:
    model_names: tuple[str, ...]
    component_temperatures: tuple[float, ...]
    blend_weights: tuple[float, ...]
    final_temperature: float
    development_logloss: float
    fit_scope: str = "DEVELOPMENT_OOS_ONLY"
    schema_version: int = 1

    def __post_init__(self) -> None:
        n = len(self.model_names)
        if n == 0 or len(self.component_temperatures) != n or len(self.blend_weights) != n:
            raise ValueError("component specification lengths must match and be non-empty")
        temps = np.asarray(self.component_temperatures, dtype=float)
        weights = np.asarray(self.blend_weights, dtype=float)
        if not np.isfinite(temps).all() or (temps <= 0).any():
            raise ValueError("component temperatures must be finite and positive")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("blend weights must be finite and non-negative")
        if not np.isclose(float(weights.sum()), 1.0, atol=1e-10):
            raise ValueError("blend weights must sum to 1")
        if not np.isfinite(self.final_temperature) or self.final_temperature <= 0:
            raise ValueError("final temperature must be finite and positive")
        if not np.isfinite(self.development_logloss):
            raise ValueError("development logloss must be finite")
        if self.fit_scope != "DEVELOPMENT_OOS_ONLY":
            raise ValueError("candidate fitter is restricted to Development OOS")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_inputs(y_true: Sequence[int], component_probabilities: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    y = np.asarray(y_true, dtype=int)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y_true must be a non-empty 1D array")
    if not component_probabilities:
        raise ValueError("at least one component is required")
    shapes = set()
    for name, values in component_probabilities.items():
        p = np.asarray(values, dtype=float)
        if p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] < 2:
            raise ValueError(f"invalid probability matrix for {name}")
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f"invalid probability values for {name}")
        row_sum = p.sum(axis=1)
        if np.any(row_sum <= 0):
            raise ValueError(f"non-positive probability row for {name}")
        shapes.add(p.shape)
    if len(shapes) != 1:
        raise ValueError("all component probability matrices must share one shape")
    n_classes = next(iter(shapes))[1]
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("y_true contains an invalid class")
    return y, n_classes


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = p.sum(axis=1, keepdims=True)
    if np.any(s <= 0) or not np.isfinite(s).all():
        raise ValueError("invalid probability mass")
    q = p / s
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("invalid normalized probabilities")
    return q


def _weight_grid(n_components: int, step: float) -> list[tuple[float, ...]]:
    if n_components <= 0 or not (0 < step <= 1):
        raise ValueError("invalid simplex grid configuration")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0, atol=1e-10):
        raise ValueError("weight step must divide 1 exactly")
    out: list[tuple[float, ...]] = []
    for counts in product(range(units + 1), repeat=n_components):
        if sum(counts) != units:
            continue
        out.append(tuple(c / units for c in counts))
    if not out:
        raise ValueError("simplex grid is empty")
    return out


def _blend(component_probs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(component_probs) != len(weights) or not component_probs:
        raise ValueError("component/weight mismatch")
    q = np.zeros_like(component_probs[0], dtype=float)
    for p, w in zip(component_probs, weights):
        q += float(w) * np.asarray(p, dtype=float)
    return _normalize(q)


def _uniform_distance(weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    u = np.full(len(w), 1.0 / len(w))
    return float(np.sum((w - u) ** 2))


def select_top_models(
    y_true: Sequence[int],
    component_probabilities: Mapping[str, Any],
    *,
    top_k: int = 3,
) -> tuple[str, ...]:
    """Pick top components by Development OOS LogLoss with deterministic ties."""
    y, _ = _validate_inputs(y_true, component_probabilities)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows = []
    for name, values in component_probabilities.items():
        p = _normalize(np.asarray(values, dtype=float))
        loss = float(log_loss(y, p, labels=list(range(p.shape[1]))))
        rows.append((loss, str(name), p))
    rows.sort(key=lambda row: (row[0], row[1]))
    return tuple(row[1] for row in rows[: min(top_k, len(rows))])


def fit_calibrated_blend(
    y_true: Sequence[int],
    component_probabilities: Mapping[str, Any],
    *,
    top_k: int = 3,
    temperature_grid: np.ndarray | None = None,
    weight_step: float = 0.25,
) -> tuple[CalibratedBlendSpec, np.ndarray]:
    """Fit an individually calibrated constrained blend on Development OOS only."""
    y, _ = _validate_inputs(y_true, component_probabilities)
    selected = select_top_models(y, component_probabilities, top_k=top_k)

    calibrated: dict[str, np.ndarray] = {}
    temperatures: dict[str, float] = {}
    for name in selected:
        raw = _normalize(np.asarray(component_probabilities[name], dtype=float))
        cal = fit_temperature(raw, y, grid=temperature_grid)
        temperatures[name] = float(cal.temperature)
        calibrated[name] = cal.transform(raw)

    names = tuple(selected)
    component_order = [calibrated[name] for name in names]
    best: tuple[float, float, int, tuple[float, ...], float, np.ndarray] | None = None
    for weights in _weight_grid(len(names), weight_step):
        blended = _blend(component_order, weights)
        final_cal = fit_temperature(blended, y, grid=temperature_grid)
        final_probs = final_cal.transform(blended)
        loss = float(log_loss(y, final_probs, labels=list(range(final_probs.shape[1]))))
        key = (loss, _uniform_distance(weights), sum(w > 0 for w in weights), weights)
        record = (*key[:3], weights, float(final_cal.temperature), final_probs)
        if best is None or record[:4] < best[:4]:
            best = record
    if best is None:
        raise RuntimeError("no constrained blend candidate was evaluated")

    loss, _, _, weights, final_temperature, final_probs = best
    spec = CalibratedBlendSpec(
        model_names=names,
        component_temperatures=tuple(temperatures[name] for name in names),
        blend_weights=tuple(float(x) for x in weights),
        final_temperature=float(final_temperature),
        development_logloss=float(loss),
    )
    return spec, _normalize(final_probs)


def apply_calibrated_blend(
    spec: CalibratedBlendSpec,
    component_probabilities: Mapping[str, Any],
) -> np.ndarray:
    """Apply a frozen Development-fitted spec; no fitting or labels are accepted."""
    parts: list[np.ndarray] = []
    for name, temperature in zip(spec.model_names, spec.component_temperatures):
        if name not in component_probabilities:
            raise KeyError(f"missing component probabilities: {name}")
        raw = _normalize(np.asarray(component_probabilities[name], dtype=float))
        parts.append(TemperatureCalibration(float(temperature)).transform(raw))
    blended = _blend(parts, spec.blend_weights)
    return _normalize(TemperatureCalibration(float(spec.final_temperature)).transform(blended))
