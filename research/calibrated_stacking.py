"""Low-dimensional individually calibrated stacking challenger.

Research-only module. It consumes strictly chronological OOS probability streams,
fits one shared-contract temperature per component on Development OOS, searches
a deterministic coarse simplex, and optionally fits one final shared temperature
to the blended Development OOS stream.

The module never reads an independent holdout and never decides production
adoption. The caller must provide provenance that each probability stream was
generated out-of-sample.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Mapping, Sequence

import numpy as np

from evaluation.calibration import fit_temperature


Array = np.ndarray


def _normalize(p: Array) -> Array:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or len(p) == 0 or p.shape[1] < 2:
        raise ValueError("probability matrix must be non-empty 2-D with >=2 classes")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0) or np.any(np.abs(sums - 1.0) > 1e-5):
        raise ValueError("probability rows must have positive sums and total 1")
    return p / sums


def _metrics(y: Array, p: Array) -> dict[str, float]:
    p = _normalize(p)
    y = np.asarray(y, dtype=int)
    if len(y) != len(p):
        raise ValueError("target/probability mismatch")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("target outside probability classes")
    idx = np.arange(len(y))
    ll = -float(np.mean(np.log(np.maximum(p[idx, y], 1e-15))))
    one = np.zeros_like(p)
    one[idx, y] = 1.0
    brier = float(np.mean(np.sum((p - one) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(p, axis=1) == y))
    return {"rows": int(len(y)), "LogLoss": ll, "Brier": brier, "Accuracy": acc}


def temperature_transform(p: Array, temperature: float) -> Array:
    if not np.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    q = np.log(np.clip(_normalize(p), 1e-12, 1.0)) / float(temperature)
    q -= q.max(axis=1, keepdims=True)
    q = np.exp(q)
    return _normalize(q)


def _simplex_weights(n_components: int, step: float) -> list[tuple[float, ...]]:
    if n_components < 2 or n_components > 3:
        raise ValueError("this challenger supports 2 or 3 components")
    if not np.isfinite(step) or step <= 0 or step > 1:
        raise ValueError("step must be in (0,1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0, atol=1e-12):
        raise ValueError("step must divide 1 exactly")
    out = []
    for grid in product(range(units + 1), repeat=n_components):
        if sum(grid) != units:
            continue
        out.append(tuple(float(x * step) for x in grid))
    # Deterministic tie-break: prefer more uniform weights, then lexical order.
    out.sort(key=lambda w: (float(np.var(w)), w))
    return out


@dataclass(frozen=True)
class CalibratedStackingSpec:
    component_names: tuple[str, ...]
    component_temperatures: dict[str, float]
    blend_weights: dict[str, float]
    final_temperature: float
    development_metrics: dict[str, float]
    search_step: float
    fit_rows: int
    candidate_type: str = "INDIVIDUAL_TEMPERATURE_COARSE_STACK"


def fit_calibrated_stacking(
    *,
    predictions: Mapping[str, Array],
    y: Array,
    component_names: Sequence[str] | None = None,
    simplex_step: float = 0.25,
    fit_final_temperature: bool = True,
) -> CalibratedStackingSpec:
    """Fit a deterministic low-dimensional stack on Development OOS only."""
    y = np.asarray(y, dtype=int)
    names = tuple(component_names or predictions.keys())
    if len(names) not in (2, 3):
        raise ValueError("exactly 2 or 3 components are required")
    if len(set(names)) != len(names):
        raise ValueError("component names must be unique")
    for name in names:
        if name not in predictions:
            raise ValueError(f"missing component prediction: {name}")
    matrices = {name: _normalize(np.asarray(predictions[name], dtype=float)) for name in names}
    rows = {len(p) for p in matrices.values()}
    classes = {p.shape[1] for p in matrices.values()}
    if len(rows) != 1 or len(classes) != 1 or next(iter(rows)) != len(y):
        raise ValueError("all components and y must have identical rows/classes")
    if len(y) < 60:
        raise ValueError("insufficient Development OOS rows for stacking")
    if len(np.unique(y)) < 2:
        raise ValueError("Development target must contain at least two classes")

    temps = {}
    calibrated = {}
    for name in names:
        fit = fit_temperature(matrices[name], y)
        temps[name] = float(fit.temperature)
        calibrated[name] = temperature_transform(matrices[name], temps[name])

    best = None
    for weights in _simplex_weights(len(names), simplex_step):
        blend = np.zeros_like(next(iter(calibrated.values())))
        for name, weight in zip(names, weights):
            blend += float(weight) * calibrated[name]
        blend = _normalize(blend)
        final_t = 1.0
        final = blend
        if fit_final_temperature:
            final_t = float(fit_temperature(blend, y).temperature)
            final = temperature_transform(blend, final_t)
        m = _metrics(y, final)
        # Primary criterion LogLoss; then Brier; then Accuracy; then simplicity.
        key = (m["LogLoss"], m["Brier"], -m["Accuracy"], float(np.var(weights)), weights)
        if best is None or key < best[0]:
            best = (key, weights, final_t, m)

    assert best is not None
    _, weights, final_t, dev_metrics = best
    return CalibratedStackingSpec(
        component_names=names,
        component_temperatures=temps,
        blend_weights={name: float(weight) for name, weight in zip(names, weights)},
        final_temperature=float(final_t),
        development_metrics=dev_metrics,
        search_step=float(simplex_step),
        fit_rows=int(len(y)),
    )


def apply_calibrated_stacking(spec: CalibratedStackingSpec, predictions: Mapping[str, Array]) -> Array:
    """Apply a frozen Development-derived stacking specification to new rows."""
    components = []
    for name in spec.component_names:
        if name not in predictions:
            raise ValueError(f"missing prediction component: {name}")
        p = temperature_transform(predictions[name], spec.component_temperatures[name])
        components.append((name, p))
    out = np.zeros_like(components[0][1])
    for name, p in components:
        out += float(spec.blend_weights[name]) * p
    out = _normalize(out)
    return temperature_transform(out, spec.final_temperature)


def spec_to_dict(spec: CalibratedStackingSpec) -> dict[str, object]:
    return asdict(spec)
