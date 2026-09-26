"""Research-only Ultimate v13 control plane.

This module adds PIT-safe control/diagnostic layers around the existing v13
foundation without mutating the production prediction path. The default E2E
runner uses a deterministic synthetic chronological fixture; synthetic success
is never treated as production performance evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import hashlib
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

from research.future_generalization_v13 import (
    FutureFailureEstimator,
    error_correlation,
    model_disagreement,
    predictability_score,
    routing_weights,
)

EPS = 1e-12


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _clip01(x: Any) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _norm_probs(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("probabilities must be a finite non-negative 2D matrix")
    sums = arr.sum(axis=1, keepdims=True)
    if np.any(sums <= EPS):
        raise ValueError("probability rows must have positive mass")
    return arr / sums


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    arr = _norm_probs(p)
    target = np.asarray(y, dtype=int).reshape(-1)
    if len(target) != len(arr):
        raise ValueError("ECE input length mismatch")
    pred = np.argmax(arr, axis=1)
    conf = arr.max(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= lo) & ((conf < hi) if i < bins - 1 else (conf <= hi))
        if mask.any():
            score += float(mask.mean()) * abs(float(conf[mask].mean()) - float((pred[mask] == target[mask]).mean()))
    return float(score)


def pit_audit(
    frame: pd.DataFrame,
    *,
    prediction_time_col: str = "prediction_time",
    available_at_col: str = "available_at",
    label_available_at_col: str | None = None,
    forbidden_future_tokens: Sequence[str] = (
        "actual", "outcome", "result", "final", "postgame", "future", "settled"
    ),
) -> dict[str, Any]:
    """Fail-closed point-in-time and structural leakage audit."""
    blockers: list[str] = []
    required = [prediction_time_col, available_at_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return {"status": "FAIL", "blockers": [f"missing:{c}" for c in missing], "rows": int(len(frame))}
    for idx, row in frame.iterrows():
        try:
            pt = _utc(row[prediction_time_col])
        except Exception:
            blockers.append(f"row_{idx}:invalid_prediction_time")
            continue
        raw = row[available_at_col]
        if pd.isna(raw) or str(raw).strip() == "":
            blockers.append(f"row_{idx}:available_at_missing")
            continue
        try:
            at = _utc(raw)
        except Exception:
            blockers.append(f"row_{idx}:invalid_available_at")
            continue
        if at > pt:
            blockers.append(f"row_{idx}:available_at_after_prediction")
        if label_available_at_col and label_available_at_col in frame.columns:
            lab = row[label_available_at_col]
            if not pd.isna(lab) and _utc(lab) <= pt:
                blockers.append(f"row_{idx}:label_available_before_or_at_prediction")
    suspicious = []
    for col in frame.columns:
        lc = col.lower()
        if any(tok in lc for tok in forbidden_future_tokens):
            if col not in {label_available_at_col, "prediction_time", "available_at"}:
                suspicious.append(col)
    warnings = [f"suspicious_column_name:{c}" for c in sorted(set(suspicious))]
    return {
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "warnings": warnings,
        "rows": int(len(frame)),
        "meta_leakage": "FAIL" if blockers else "PASS",
    }


def data_quality(frame: pd.DataFrame, *, key_cols: Sequence[str] = ()) -> dict[str, Any]:
    if frame.empty:
        return {"status": "FAIL", "score": 0.0, "missing_rate": 1.0, "duplicate_rate": 1.0}
    missing_rate = float(frame.isna().mean().mean())
    duplicate_rate = float(frame.duplicated(list(key_cols) if key_cols else None).mean())
    numeric = frame.select_dtypes(include=[np.number])
    nonfinite_rate = float((~np.isfinite(numeric.to_numpy(dtype=float))).mean()) if not numeric.empty else 0.0
    score = np.clip(1.0 - 0.45 * missing_rate - 0.30 * duplicate_rate - 0.25 * nonfinite_rate, 0.0, 1.0)
    return {
        "status": "PASS" if score >= 0.80 else "WARN",
        "score": float(score),
        "missing_rate": missing_rate,
        "duplicate_rate": duplicate_rate,
        "nonfinite_rate": nonfinite_rate,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
    }


def feature_reliability(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in features:
        if col not in frame.columns:
            out[col] = 0.0
            continue
        s = frame[col]
        completeness = float(s.notna().mean())
        arr = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
        finite = float(np.isfinite(arr).mean()) if len(arr) else 0.0
        variability = 1.0 if len(arr) < 3 else float(np.clip(np.std(arr) / max(abs(np.mean(arr)), 1.0), 0.0, 1.0))
        stability = 1.0 - variability * 0.35
        out[col] = float(np.clip(0.45 * completeness + 0.35 * finite + 0.20 * stability, 0.0, 1.0))
    return out


def source_reliability(*, success_rate: float, freshness: float, consistency: float, latency: float) -> float:
    return float(np.clip(0.35 * success_rate + 0.25 * freshness + 0.25 * consistency + 0.15 * (1.0 - np.clip(latency, 0.0, 1.0)), 0.0, 1.0))


def drift_score(reference: pd.DataFrame, current: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    per_feature: dict[str, float] = {}
    for col in columns:
        if col not in reference or col not in current:
            continue
        a = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy(float)
        if len(a) < 3 or len(b) < 3:
            continue
        mu, sd = float(np.mean(a)), float(np.std(a) + EPS)
        per_feature[col] = float(np.clip(abs(np.mean(b) - mu) / (sd + EPS) / 3.0, 0.0, 1.0))
    score = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
    return {"score": score, "per_feature": per_feature, "status": "WARN" if score >= 0.33 else "PASS"}


def ood_score(reference: pd.DataFrame, row: Mapping[str, Any], columns: Sequence[str]) -> float:
    vals = []
    for col in columns:
        if col not in reference or col not in row:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy(float)
        if len(ref) < 5:
            continue
        x = float(row[col])
        mu, sd = float(np.mean(ref)), float(np.std(ref) + EPS)
        vals.append(abs(x - mu) / sd)
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals) / 5.0, 0.0, 1.0))


def regime_state(*, volatility: float, trend: float, information_shock: float) -> str:
    if information_shock >= 0.75:
        return "information_shock"
    if volatility >= 0.75 and abs(trend) >= 0.20:
        return "high_vol_trend"
    if volatility >= 0.75:
        return "high_volatility"
    if abs(trend) >= 0.20:
        return "trend"
    if volatility <= 0.20:
        return "low_volatility"
    return "range"


def regime_transition(history_states: Sequence[str], *, horizon: int = 1) -> dict[str, Any]:
    if not history_states:
        return {"current": "unknown", "transition": {"unknown": 1.0}, "horizon": horizon}
    current = str(history_states[-1])
    transitions: dict[str, dict[str, int]] = {}
    for a, b in zip(history_states[:-1], history_states[1:]):
        transitions.setdefault(a, {})[b] = transitions.setdefault(a, {}).get(b, 0) + 1
    counts = transitions.get(current, {current: 1})
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    for _ in range(max(0, horizon - 1)):
        next_probs: dict[str, float] = {}
        for src, srcp in probs.items():
            row = transitions.get(src, {src: 1})
            denom = sum(row.values())
            for dst, cnt in row.items():
                next_probs[dst] = next_probs.get(dst, 0.0) + srcp * cnt / denom
        probs = next_probs or {current: 1.0}
    return {"current": current, "transition": probs, "horizon": horizon}


def retrieval(
    history: pd.DataFrame,
    query: Mapping[str, float],
    *,
    feature_cols: Sequence[str],
    prediction_time: Any,
    k: int = 5,
) -> dict[str, Any]:
    """Retrieve only historical rows that were available before prediction time."""
    pt = _utc(prediction_time)
    if "prediction_time" not in history.columns or "available_at" not in history.columns:
        return {"status": "BLOCKED", "reason": "history missing PIT columns", "neighbors": []}
    candidates = history.copy()
    candidates["_pt"] = pd.to_datetime(candidates["prediction_time"], utc=True, errors="coerce")
    candidates["_at"] = pd.to_datetime(candidates["available_at"], utc=True, errors="coerce")
    candidates = candidates[(candidates["_pt"] < pt) & (candidates["_at"] <= pt)]
    rows = []
    for idx, row in candidates.iterrows():
        dist = 0.0
        used = 0
        for col in feature_cols:
            if col not in row or col not in query or pd.isna(row[col]) or pd.isna(query[col]):
                continue
            dist += (float(row[col]) - float(query[col])) ** 2
            used += 1
        if used:
            rows.append((math.sqrt(dist / used), idx))
    rows.sort(key=lambda x: (x[0], str(x[1])))
    selected = [idx for _, idx in rows[: max(1, int(k))]]
    return {
        "status": "PASS",
        "neighbors": [{"index": str(i), "distance": float(d)} for d, i in rows[: max(1, int(k))]],
        "retrieved_rows": int(len(selected)),
        "pit_filtered_rows": int(len(candidates)),
    }


def meta_label_walk_forward(y: Sequence[int], p: Sequence[float], *, min_train: int = 40) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    if len(y_arr) != len(p_arr):
        raise ValueError("meta label input mismatch")
    correct = (p_arr >= 0.5).astype(int) == y_arr
    labels = correct.astype(int)
    probs = np.full(len(y_arr), np.nan, dtype=float)
    for t in range(max(1, int(min_train)), len(y_arr)):
        x = np.column_stack([p_arr[:t], np.abs(p_arr[:t] - 0.5)])
        if len(np.unique(labels[:t])) < 2:
            continue
        model = LogisticRegression(C=0.5, max_iter=200, random_state=42)
        model.fit(x, labels[:t])
        xt = np.array([[p_arr[t], abs(p_arr[t] - 0.5)]], dtype=float)
        probs[t] = float(model.predict_proba(xt)[0, 1])
    usable = np.isfinite(probs)
    return {
        "status": "PASS" if usable.any() else "BLOCKED",
        "probs": probs,
        "coverage": float(usable.mean()),
        "mean_meta_prob": float(np.nanmean(probs)) if usable.any() else None,
        "label_is_future_free": True,
    }


def uncertainty_decomposition(*, disagreement: float, ood: float, drift: float, data_quality_score: float, info_uncertainty: float) -> dict[str, float]:
    data = _clip01(1.0 - data_quality_score)
    model = _clip01(disagreement)
    distribution = _clip01(0.6 * ood + 0.4 * drift)
    information = _clip01(info_uncertainty)
    aleatoric = _clip01(0.5 * (1.0 - data_quality_score) + 0.5 * information)
    total = _clip01(0.30 * data + 0.25 * model + 0.20 * distribution + 0.15 * information + 0.10 * aleatoric)
    return {"data": data, "model": model, "distribution": distribution, "information": information, "aleatoric": aleatoric, "total": total}


def active_information(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = []
    for row in candidates:
        name = str(row["name"])
        gain = float(row.get("expected_gain", 0.0))
        cost = float(row.get("cost", 0.0))
        risk = float(row.get("failure_risk", 0.0))
        value = gain - cost - 0.5 * risk
        scored.append({"name": name, "value": value, "gain": gain, "cost": cost, "risk": risk})
    scored.sort(key=lambda x: (x["value"], x["name"]), reverse=True)
    chosen = scored[0] if scored and scored[0]["value"] > 0 else None
    return {"selected": chosen["name"] if chosen else None, "scores": scored, "expected_net_value": float(chosen["value"] if chosen else 0.0)}


def adaptive_compute(*, predictability: float, uncertainty: float, failure_risk: float, ood: float) -> dict[str, Any]:
    difficulty = float(np.clip(0.35 * (1.0 - predictability) + 0.25 * uncertainty + 0.25 * failure_risk + 0.15 * ood, 0.0, 1.0))
    if ood >= 0.90 or uncertainty >= 0.92:
        level = "ABSTAIN"
    elif difficulty >= 0.70:
        level = "DEEP_COMPUTE"
    elif difficulty >= 0.45:
        level = "ENSEMBLE_RETRIEVAL"
    else:
        level = "STANDARD"
    budget = {"STANDARD": 1.0, "ENSEMBLE_RETRIEVAL": 2.0, "DEEP_COMPUTE": 4.0, "ABSTAIN": 0.5}[level]
    return {"difficulty": difficulty, "compute_level": level, "budget_units": budget}


def prediction_trajectory(current_p: np.ndarray, *, momentum: float, horizons: Sequence[int]) -> list[dict[str, Any]]:
    p = _norm_probs(np.asarray(current_p))[0]
    out = []
    for h in horizons:
        drift = float(momentum) * (1.0 - math.exp(-0.18 * float(h)))
        q = p.copy()
        if len(q) >= 2:
            q[0] += drift
            q[-1] -= drift
        q = np.clip(q, 1e-6, None)
        q /= q.sum()
        out.append({"horizon": int(h), "probabilities": [float(x) for x in q]})
    return out


def revision_policy(previous: np.ndarray, current: np.ndarray, *, stability_budget: int, revisions_used: int) -> dict[str, Any]:
    old = _norm_probs(previous)[0]
    new = _norm_probs(current)[0]
    shift = float(0.5 * np.abs(old - new).sum())
    if revisions_used >= stability_budget and shift < 0.15:
        action = "MAINTAIN"
    elif shift < 0.03:
        action = "MAINTAIN"
    elif shift < 0.12:
        action = "MINOR_REVISION"
    else:
        action = "MAJOR_REVISION"
    return {"action": action, "revision_size": shift, "revision_required": action != "MAINTAIN"}


def selective_prediction(y: Sequence[int], p: np.ndarray, *, risk_score: Sequence[float], threshold: float = 0.65) -> dict[str, Any]:
    arr = _norm_probs(p)
    target = np.asarray(y, dtype=int)
    risk = np.asarray(risk_score, dtype=float)
    accepted = risk < float(threshold)
    pred = np.argmax(arr, axis=1)
    coverage = float(accepted.mean())
    accuracy = float((pred[accepted] == target[accepted]).mean()) if accepted.any() else float("nan")
    return {"coverage": coverage, "selective_accuracy": accuracy, "threshold": float(threshold), "accepted": accepted.astype(int).tolist()}


def robustness_matrix(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    arr = _norm_probs(p)
    tests: dict[str, Any] = {}
    base_pred = np.argmax(arr, axis=1)
    tests["baseline"] = {"accuracy": float((base_pred == y).mean())}
    noisy = np.clip(arr + 0.03 * np.sin(np.arange(arr.shape[0])[:, None]), 1e-6, None)
    noisy /= noisy.sum(axis=1, keepdims=True)
    tests["feature_noise"] = {"accuracy": float((np.argmax(noisy, axis=1) == y).mean())}
    dropped = arr.copy()
    if dropped.shape[1] >= 3:
        dropped[:, 1] = dropped[:, 1] * 0.5
    dropped /= dropped.sum(axis=1, keepdims=True)
    tests["feature_drop_proxy"] = {"accuracy": float((np.argmax(dropped, axis=1) == y).mean())}
    flipped = np.roll(arr, 1, axis=1)
    tests["prediction_shock"] = {"accuracy": float((np.argmax(flipped, axis=1) == y).mean())}
    worst = min(v["accuracy"] for v in tests.values())
    return {"tests": tests, "worst_accuracy": float(worst), "stability_drop": float(tests["baseline"]["accuracy"] - worst)}


def ablation_summary(y: np.ndarray, baseline: np.ndarray, variants: Mapping[str, np.ndarray]) -> dict[str, Any]:
    def metric(p):
        arr = _norm_probs(p)
        pred = np.argmax(arr, axis=1)
        ll = log_loss(y, arr, labels=list(range(arr.shape[1])))
        if arr.shape[1] == 2:
            br = brier_score_loss(y, arr[:, 1])
        else:
            one = np.eye(arr.shape[1])[y]
            br = float(np.mean(np.sum((arr - one) ** 2, axis=1)))
        return {"Accuracy": float((pred == y).mean()), "LogLoss": float(ll), "Brier": float(br), "ECE": _ece(y, arr)}
    base = metric(baseline)
    out = {"baseline": base, "variants": {name: metric(p) for name, p in variants.items()}}
    for name, m in out["variants"].items():
        m["delta_Accuracy"] = m["Accuracy"] - base["Accuracy"]
        m["delta_LogLoss"] = m["LogLoss"] - base["LogLoss"]
        m["delta_Brier"] = m["Brier"] - base["Brier"]
        m["delta_ECE"] = m["ECE"] - base["ECE"]
    return out


@dataclass(frozen=True)
class ControlStatus:
    component: str
    state: str
    evidence_scope: str
    note: str = ""


def artifact_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "evidence_scope", "pit", "data_quality", "disagreement",
        "predictability", "future_failure", "routing", "uncertainty", "ood",
        "regime", "retrieval", "meta_label", "active_information", "adaptive_compute",
        "trajectory", "revision", "selective", "ablation", "robustness",
    }
    missing = sorted(required - set(report))
    return {"status": "PASS" if not missing else "FAIL", "missing": missing}


def promotion_gate(*, report: Mapping[str, Any], pit: bool, leakage: bool, oos: bool, robustness: bool, calibration: bool, reproducible: bool) -> dict[str, Any]:
    blockers = []
    if not pit: blockers.append("PIT_FAIL")
    if not leakage: blockers.append("LEAKAGE_FAIL")
    if not oos: blockers.append("OOS_NOT_VERIFIED")
    if not robustness: blockers.append("ROBUSTNESS_FAIL")
    if not calibration: blockers.append("CALIBRATION_FAIL")
    if not reproducible: blockers.append("REPRODUCIBILITY_FAIL")
    if report.get("evidence_scope") != "real_oos":
        blockers.append("REAL_OOS_EVIDENCE_REQUIRED")
    return {
        "status": "PASS" if not blockers else "HOLD",
        "promotion_allowed": not blockers,
        "blockers": blockers,
        "auto_promotion": False,
    }


def _fixture(seed: int = 13, n: int = 360, k: int = 3) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2025-01-01T00:00:00Z")
    states = ["range", "trend", "high_volatility", "low_volatility", "information_shock"]
    rows = []
    y = rng.integers(0, k, size=n)
    for i in range(n):
        pt = base_time + pd.Timedelta(hours=i)
        rows.append({
            "prediction_time": pt,
            "available_at": pt - pd.Timedelta(minutes=45),
            "feature_a": float(np.sin(i / 13.0) + rng.normal(0, 0.10)),
            "feature_b": float(np.cos(i / 17.0) + rng.normal(0, 0.12)),
            "volatility": float(np.clip(0.45 + 0.35 * np.sin(i / 29.0) + rng.normal(0, 0.06), 0, 1)),
            "trend": float(0.28 * np.sin(i / 35.0) + rng.normal(0, 0.05)),
            "information_shock": float(1.0 if i % 83 == 0 and i else 0.05),
            "regime": states[(i // 72) % len(states)],
        })
    frame = pd.DataFrame(rows)
    models: dict[str, np.ndarray] = {}
    losses: dict[str, list[float]] = {}
    onehot = np.eye(k)[y]
    for j, noise in enumerate((0.015, 0.035, 0.060)):
        raw = 0.18 + 0.65 * onehot
        p = np.clip(raw + rng.normal(0, noise, raw.shape), 1e-4, None)
        p /= p.sum(axis=1, keepdims=True)
        models[f"M{j}"] = p
        losses[f"M{j}"] = (-np.log(np.clip(p[np.arange(n), y], EPS, 1))).tolist()
    return frame, y, models, losses


def run_control_plane() -> dict[str, Any]:
    frame, y, models, losses = _fixture()
    split = 260
    ref = frame.iloc[:split].copy()
    cur = frame.iloc[split:].copy()
    pt = cur.iloc[0]["prediction_time"]
    history_ref = frame.iloc[: split - 60].copy()
    recent_ref = frame.iloc[split - 60 : split].copy()
    pit = pit_audit(frame[["prediction_time", "available_at"]])
    dq = data_quality(frame, key_cols=("prediction_time",))
    rel = feature_reliability(frame, ("feature_a", "feature_b", "volatility", "trend"))
    source_rel = source_reliability(success_rate=0.99, freshness=0.96, consistency=0.93, latency=0.04)
    drift = drift_score(history_ref, recent_ref, ("feature_a", "feature_b", "volatility", "trend"))
    row = cur.iloc[0]
    ood = ood_score(ref, row, ("feature_a", "feature_b", "volatility", "trend"))
    d = model_disagreement(models)
    errcorr = error_correlation(y[split:], {k: v[split:] for k, v in models.items()})
    mean_p = d["mean_probability"]
    pred = predictability_score(mean_p[:split], history_probs=mean_p[:split], data_quality=dq["score"], drift_score=drift["score"])
    train_losses = {k: v[:split] for k, v in losses.items()}
    failure = FutureFailureEstimator(horizon=8).fit(train_losses).predict(train_losses)
    max_risk = max(v["failure_risk"] for v in failure.values())
    recent_losses = {k: float(np.mean(v[-12:])) for k, v in train_losses.items()}
    boundary_disagreement = model_disagreement({k: v[split-1:split] for k, v in models.items()})
    weights = routing_weights(recent_losses, failure_risk={k: v["failure_risk"] for k, v in failure.items()}, predictability=pred["predictability"], disagreement=float(boundary_disagreement["disagreement_score"][-1]))
    routed_p = np.sum(np.stack([models[n] for n in weights], axis=1) * np.array(list(weights.values()))[None, :, None], axis=1)
    final_p = routed_p[split]
    regime = {"current": frame["regime"].iloc[split-1], "transition": regime_transition(frame["regime"].iloc[:split].tolist(), horizon=3)}
    retr = retrieval(ref, {c: float(row[c]) for c in ("feature_a", "feature_b", "volatility", "trend")}, feature_cols=("feature_a", "feature_b", "volatility", "trend"), prediction_time=pt, k=8)
    meta = meta_label_walk_forward(y[:split], models["M0"][:split, 0], min_train=60)
    unc = uncertainty_decomposition(disagreement=float(boundary_disagreement["disagreement_score"][-1]), ood=ood, drift=drift["score"], data_quality_score=dq["score"], info_uncertainty=0.14)
    info = active_information([
        {"name": "starter_update", "expected_gain": 0.08, "cost": 0.02, "failure_risk": 0.01},
        {"name": "weather_refresh", "expected_gain": 0.03, "cost": 0.02, "failure_risk": 0.03},
        {"name": "market_refresh", "expected_gain": 0.05, "cost": 0.05, "failure_risk": 0.02},
    ])
    compute = adaptive_compute(predictability=pred["predictability"], uncertainty=unc["total"], failure_risk=max_risk, ood=ood)
    traj = prediction_trajectory(final_p, momentum=0.04 * (1.0 if row["trend"] >= 0 else -1.0), horizons=(1, 2, 4, 8))
    revised = revision_policy(routed_p[split-1], final_p, stability_budget=3, revisions_used=0)
    risk = 0.45 * np.full(len(cur), unc["total"]) + 0.55 * (1.0 - mean_p[split:].max(axis=1))
    selective = selective_prediction(y[split:], mean_p[split:], risk_score=risk, threshold=0.65)
    alt = final_p.copy(); alt[0] = max(1e-6, alt[0] - 0.03); alt /= alt.sum()
    counterfactual = {"total_variation": float(0.5 * np.abs(final_p - alt).sum()), "stable": bool(0.5 * np.abs(final_p - alt).sum() < 0.10)}
    ablate = ablation_summary(y[split:], mean_p[split:], {"routed": routed_p[split:]})
    ablate["coverage"] = {"implemented_variants": 1, "required_matrix_variants": 14, "status": "PARTIAL"}
    robust = robustness_matrix(y[split:], mean_p[split:])
    oos_metrics = ablate["baseline"]
    routed_metrics = ablate["variants"]["routed"]
    states_list = [
        ControlStatus("PIT", "EXECUTED" if pit["status"] == "PASS" else "FAILED", "synthetic_pit_fixture"),
        ControlStatus("DataQuality", "EXECUTED", "synthetic_pit_fixture"),
        ControlStatus("FeatureReliability", "EXECUTED", "synthetic_pit_fixture"),
        ControlStatus("ModelDisagreement", "EXECUTED", "synthetic_oos_fixture"),
        ControlStatus("Predictability", "EXECUTED", "synthetic_oos_fixture"),
        ControlStatus("FutureFailure", "EXECUTED", "synthetic_oos_fixture"),
        ControlStatus("TimeToFailure", "EXECUTED", "synthetic_oos_fixture"),
        ControlStatus("RegimeTransition", "EXECUTED", "synthetic_oos_fixture"),
        ControlStatus("Retrieval", "EXECUTED", "synthetic_pit_fixture"),
        ControlStatus("MetaLabel", "EXECUTED", "synthetic_walk_forward"),
        ControlStatus("Uncertainty", "EXECUTED", "synthetic_oos_fixture"),
        ControlStatus("ActiveInformation", "EXECUTED", "synthetic_control_fixture"),
        ControlStatus("AdaptiveCompute", "EXECUTED", "synthetic_control_fixture"),
        ControlStatus("PredictionTrajectory", "EXECUTED", "synthetic_control_fixture"),
        ControlStatus("RevisionPolicy", "EXECUTED", "synthetic_control_fixture"),
        ControlStatus("SelectivePrediction", "EXECUTED", "synthetic_oos_fixture"),
        ControlStatus("Ablation", "EXECUTED", "synthetic_oos_fixture", "partial_implemented_matrix"),
        ControlStatus("Robustness", "EXECUTED", "synthetic_stress_fixture", "proxy_stress_only"),
    ]
    report = {
        "schema_version": 1,
        "evidence_scope": "synthetic_oos",
        "promotion_status": "HOLD",
        "oos_scope": {"train_rows": int(split), "oos_rows": int(len(cur)), "chronological": True},
        "synthetic_oos_metrics": {"baseline": oos_metrics, "routed": routed_metrics},
        "pit": pit,
        "data_quality": dq,
        "feature_reliability": rel,
        "source_reliability": source_rel,
        "drift": drift,
        "ood": ood,
        "disagreement": {"last": float(d["disagreement_score"][-1]), "mean": float(np.mean(d["disagreement_score"])), "boundary": float(boundary_disagreement["disagreement_score"][-1])},
        "error_correlation": errcorr,
        "predictability": pred,
        "future_failure": failure,
        "routing": weights,
        "regime": regime,
        "retrieval": retr,
        "meta_label": {k: v for k, v in meta.items() if k != "probs"},
        "uncertainty": unc,
        "active_information": info,
        "adaptive_compute": compute,
        "trajectory": traj,
        "revision": revised,
        "selective": {k: v for k, v in selective.items() if k != "accepted"},
        "ablation": ablate,
        "robustness": robust,
        "counterfactual_stability": counterfactual,
        "calibration": {"ECE": float(_ece(y[split:], mean_p[split:]))},
        "time_to_failure": {k: float(v["time_to_failure_periods"]) for k, v in failure.items()},
        "strategy_failure": {"status": "available", "scope": "research_only"},
        "states": [asdict(s) for s in states_list],
        "artifact_gate": {},
        "promotion_gate": {},
    }
    report["artifact_gate"] = artifact_gate(report)
    report["promotion_gate"] = promotion_gate(report=report, pit=True, leakage=True, oos=False, robustness=robust["stability_drop"] <= 0.15, calibration=True, reproducible=True)
    payload = repr(sorted((k, str(v)) for k, v in report.items())).encode()
    report["report_fingerprint"] = hashlib.sha256(payload).hexdigest()
    return report
