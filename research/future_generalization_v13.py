"""Ultimate Future-Generalization v13 research controller.

Research-only utilities. Production artifacts are never mutated here.
Supervised inputs must be supplied from strictly prior, PIT-safe history by
the caller. Unknown PIT is fail-closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-12


def _utc(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _norm(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("probabilities must be finite, non-negative, and 2D")
    s = x.sum(axis=1, keepdims=True)
    if np.any(s <= EPS):
        raise ValueError("probability rows must have positive mass")
    return x / s


def _softmax(z: np.ndarray) -> np.ndarray:
    x = np.asarray(z, dtype=float)
    x = x - np.max(x)
    e = np.exp(np.clip(x, -60.0, 60.0))
    return e / max(float(e.sum()), EPS)


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return max(
        0.0,
        0.5 * float(np.sum(p * np.log(np.clip(p / np.clip(m, EPS, None), EPS, None)))),
        0.0,
    ) + 0.5 * float(np.sum(q * np.log(np.clip(q / np.clip(m, EPS, None), EPS, None))))


def _cosine_distance(p: np.ndarray, q: np.ndarray) -> float:
    den = float(np.linalg.norm(p) * np.linalg.norm(q))
    return 0.0 if den <= EPS else float(1.0 - np.dot(p, q) / den)


def error_correlation(
    y_true: np.ndarray,
    model_probs: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Historical OOS error dependence; targets are required explicitly."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    names = tuple(model_probs.keys())
    if len(names) < 2:
        raise ValueError("at least two models are required")
    ps = [_norm(model_probs[name]) for name in names]
    if any(len(p) != len(y) for p in ps):
        raise ValueError("y_true and model probability lengths differ")
    errs = np.column_stack([(np.argmax(p, axis=1) != y).astype(float) for p in ps])
    corr = np.corrcoef(errs, rowvar=False)
    if not np.all(np.isfinite(corr)):
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
    return {
        "models": list(names),
        "pairwise_error_correlation": {
            f"{names[i]}::{names[j]}": float(corr[i, j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        },
        "error_overlap": {
            f"{names[i]}::{names[j]}": float(np.mean((errs[:, i] == 1) & (errs[:, j] == 1)))
            for i in range(len(names))
            for j in range(i + 1, len(names))
        },
        "mean_error_rate": {
            name: float(errs[:, i].mean()) for i, name in enumerate(names)
        },
    }


def model_disagreement(model_probs: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Distribution and class disagreement without targets."""
    names = tuple(model_probs.keys())
    if len(names) < 2:
        raise ValueError("at least two models are required")
    ps = [_norm(model_probs[name]) for name in names]
    n, k = len(ps[0]), ps[0].shape[1]
    if any(len(p) != n or p.shape[1] != k for p in ps):
        raise ValueError("model probability shapes differ")
    stack = np.stack(ps, axis=1)
    mean_p = stack.mean(axis=1)
    top = np.argmax(stack, axis=2)
    mean_top = np.argmax(mean_p, axis=1)
    agreement = np.mean(top == mean_top[:, None], axis=1)
    entropy = -np.sum(mean_p * np.log(np.clip(mean_p, EPS, 1.0)), axis=1)
    pairwise = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            l1 = 0.5 * np.abs(ps[i] - ps[j]).sum(axis=1)
            l2 = np.linalg.norm(ps[i] - ps[j], axis=1)
            js = np.array([_js(ps[i][r], ps[j][r]) for r in range(n)])
            cosine = np.array([_cosine_distance(ps[i][r], ps[j][r]) for r in range(n)])
            kl = np.array([
                np.sum(ps[i][r] * np.log(
                    np.clip(ps[i][r] / np.clip(ps[j][r], EPS, None), EPS, None)
                ))
                for r in range(n)
            ])
            pairwise.append({
                "pair": f"{a}::{b}",
                "l1": float(l1.mean()),
                "l2": float(l2.mean()),
                "js": float(js.mean()),
                "cosine_distance": float(cosine.mean()),
                "kl_p_to_q": float(kl.mean()),
            })
    return {
        "model_count": len(names),
        "models": list(names),
        "mean_probability": mean_p,
        "probability_std": stack.std(axis=1).mean(axis=1),
        "probability_range": stack.max(axis=1).max(axis=1) - stack.min(axis=1).min(axis=1),
        "class_agreement": agreement,
        "disagreement_score": 1.0 - agreement,
        "entropy": entropy,
        "normalized_entropy": entropy / max(float(np.log(k)), EPS),
        "pairwise": pairwise,
        "mean_pairwise_js": float(np.mean([x["js"] for x in pairwise])),
        "mean_pairwise_l1": float(np.mean([x["l1"] for x in pairwise])),
    }


def predictability_score(
    current_probs: np.ndarray,
    *,
    history_probs: np.ndarray | None = None,
    data_quality: float = 1.0,
    regime_stability: float = 1.0,
    drift_score: float = 0.0,
    information_uncertainty: float = 0.0,
) -> dict[str, float]:
    """Estimate predictability separately from model confidence."""
    p = _norm(current_probs)
    current = p[-1]
    confidence = float(current.max())
    entropy = float(-np.sum(current * np.log(np.clip(current, EPS, 1.0))))
    entropy_norm = entropy / max(float(np.log(p.shape[1])), EPS)
    if history_probs is None or len(history_probs) < 2:
        temporal_stability = 0.5
    else:
        h = _norm(history_probs)
        changes = 0.5 * np.abs(h[1:] - h[:-1]).sum(axis=1)
        temporal_stability = float(np.clip(1.0 - changes.mean(), 0.0, 1.0))
    score = (
        0.30 * (1.0 - entropy_norm)
        + 0.15 * confidence
        + 0.20 * temporal_stability
        + 0.15 * np.clip(float(data_quality), 0.0, 1.0)
        + 0.10 * np.clip(float(regime_stability), 0.0, 1.0)
        + 0.05 * (1.0 - np.clip(float(drift_score), 0.0, 1.0))
        + 0.05 * (1.0 - np.clip(float(information_uncertainty), 0.0, 1.0))
    )
    return {
        "predictability": float(np.clip(score, 0.0, 1.0)),
        "confidence": confidence,
        "entropy": entropy,
        "temporal_stability": temporal_stability,
    }


class FutureFailureEstimator:
    """Research-only chronological model-level future failure estimator."""

    def __init__(self, horizon: int = 8):
        self.horizon = max(1, int(horizon))
        self.models: dict[str, LogisticRegression | None] = {}
        self.fallback: dict[str, float] = {}
        self.thresholds: dict[str, float] = {}

    @staticmethod
    def _features(losses: np.ndarray, t: int) -> np.ndarray:
        hist = losses[: t + 1]
        recent = hist[-min(12, len(hist)):]
        slope = 0.0
        if len(recent) >= 3:
            slope = float(np.polyfit(np.arange(len(recent), dtype=float), recent, 1)[0])
        return np.array(
            [hist[-1], recent.mean(), recent.std(), slope],
            dtype=float,
        )

    def fit(self, loss_history: Mapping[str, Sequence[float]]) -> "FutureFailureEstimator":
        for name, raw in loss_history.items():
            losses = np.asarray(raw, dtype=float).reshape(-1)
            if len(losses) < self.horizon + 24 or not np.all(np.isfinite(losses)):
                raise ValueError(f"invalid loss history for {name}")
            threshold = float(np.quantile(losses, 0.75))
            rows, labels = [], []
            for t in range(12, len(losses) - self.horizon):
                current = float(losses[max(0, t - 11):t + 1].mean())
                future = float(losses[t + 1:t + 1 + self.horizon].mean())
                rows.append(self._features(losses, t))
                labels.append(int(future > max(threshold, current * 1.12)))
            x, y = np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)
            self.thresholds[name] = threshold
            self.fallback[name] = float(y.mean()) if len(y) else 0.0
            if len(np.unique(y)) < 2:
                self.models[name] = None
            else:
                model = LogisticRegression(C=0.5, max_iter=500, random_state=42)
                model.fit(x, y)
                self.models[name] = model
        return self

    def predict(self, loss_history: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
        out = {}
        for name, model in self.models.items():
            hist = np.asarray(loss_history[name], dtype=float).reshape(-1)
            if len(hist) < 12 or not np.all(np.isfinite(hist)):
                raise ValueError(f"invalid loss history for {name}")
            feat = self._features(hist, len(hist) - 1).reshape(1, -1)
            risk = self.fallback[name] if model is None else float(model.predict_proba(feat)[0, 1])
            risk = float(np.clip(risk, 0.0, 1.0))
            hazard = max(risk, EPS)
            ttf = np.log(0.5) / np.log(max(1.0 - hazard, EPS))
            if not np.isfinite(ttf):
                ttf = float(self.horizon * 4)
            out[name] = {
                "failure_risk": risk,
                "hazard": risk,
                "survival_1": 1.0 - risk,
                "time_to_failure_periods": float(np.clip(ttf, 1.0, self.horizon * 4.0)),
            }
        return out


def routing_weights(
    global_losses: Mapping[str, float],
    *,
    failure_risk: Mapping[str, float],
    predictability: float,
    disagreement: float,
    future_regime_fit: Mapping[str, float] | None = None,
    concentration: float = 3.0,
) -> dict[str, float]:
    """Soft routing with future-risk penalties and no hard switch."""
    names = list(global_losses)
    if not names:
        raise ValueError("global_losses is empty")
    scores = []
    for name in names:
        loss = float(global_losses[name])
        risk = float(np.clip(failure_risk.get(name, 0.0), 0.0, 1.0))
        regime_bonus = float((future_regime_fit or {}).get(name, 0.0))
        if not np.isfinite(loss) or loss <= 0:
            raise ValueError(f"invalid loss for {name}")
        scores.append(-loss - 1.5 * risk + 0.35 * regime_bonus)
    temp = 0.35 + 0.9 * (1.0 - np.clip(float(predictability), 0.0, 1.0))
    if float(disagreement) > 0.35:
        temp += 0.35
    scaled = np.asarray(scores) / max(temp * max(float(concentration), EPS), EPS)
    w = _softmax(scaled)
    floor = min(0.04 / len(names), 0.25 / len(names))
    w = floor + (1.0 - floor * len(names)) * w
    w /= w.sum()
    return {name: float(v) for name, v in zip(names, w)}


class PredictionAction(str, Enum):
    MAINTAIN = "MAINTAIN"
    MINOR_REVISION = "MINOR_REVISION"
    MAJOR_REVISION = "MAJOR_REVISION"
    RECOMPUTE = "RECOMPUTE"
    ADD_INFORMATION = "ADD_INFORMATION"
    RETRIEVAL = "RETRIEVAL"
    SCENARIO = "SCENARIO"
    DEEP_COMPUTE = "DEEP_COMPUTE"
    ABSTAIN = "ABSTAIN"
    FALLBACK = "FALLBACK"


def select_prediction_policy(
    *,
    predictability: float,
    ood_score: float,
    failure_risk_max: float,
    information_value: float = 0.0,
    tail_risk: float = 0.0,
) -> str:
    if float(ood_score) >= 0.85 or float(tail_risk) >= 0.90:
        return PredictionAction.ABSTAIN.value
    if float(predictability) <= 0.20:
        return PredictionAction.SCENARIO.value
    if float(failure_risk_max) >= 0.75:
        return PredictionAction.DEEP_COMPUTE.value
    if float(information_value) >= 0.05:
        return PredictionAction.ADD_INFORMATION.value
    if float(predictability) <= 0.42:
        return PredictionAction.RETRIEVAL.value
    return PredictionAction.RECOMPUTE.value


def choose_information(
    candidates: Sequence[Mapping[str, float]],
) -> dict[str, float | str | None]:
    if not candidates:
        return {"selected": None, "score": 0.0}
    ranked = []
    for row in candidates:
        name = str(row["name"])
        gain, cost, risk = (
            float(row.get("expected_gain", 0.0)),
            float(row.get("cost", 0.0)),
            float(row.get("failure_risk", 0.0)),
        )
        if not all(np.isfinite(v) for v in (gain, cost, risk)):
            raise ValueError(f"invalid information candidate: {name}")
        ranked.append((gain - cost - 0.5 * risk, name))
    score, name = max(ranked)
    return {"selected": name, "score": float(max(score, 0.0))}


def decide_revision(
    previous_probs: np.ndarray | None,
    new_probs: np.ndarray,
    *,
    information_value: float = 0.0,
    max_revision_count: int = 3,
    revision_count: int = 0,
) -> dict[str, object]:
    newp = _norm(np.asarray(new_probs))
    if previous_probs is None:
        return {"action": PredictionAction.RECOMPUTE.value, "revision_size": 1.0, "revision_required": True}
    oldp = _norm(np.asarray(previous_probs))
    if oldp.shape != newp.shape or len(newp) != 1:
        raise ValueError("revision probability shape mismatch")
    shift = float(0.5 * np.abs(oldp[0] - newp[0]).sum())
    if revision_count >= max_revision_count and information_value < 0.10:
        action = PredictionAction.MAINTAIN.value
    elif shift < 0.03 and information_value < 0.03:
        action = PredictionAction.MAINTAIN.value
    elif shift < 0.12:
        action = PredictionAction.MINOR_REVISION.value
    else:
        action = PredictionAction.MAJOR_REVISION.value
    return {
        "action": action,
        "revision_size": shift,
        "revision_required": action != PredictionAction.MAINTAIN.value,
    }


@dataclass(frozen=True)
class ForecastContract:
    prediction_time: str
    valid_until: str
    model_version: str
    strategy: str
    confidence: float
    predictability: float
    uncertainty: float
    pit_status: str


@dataclass(frozen=True)
class LedgerRecord:
    prediction_time: str
    strategy: str
    action: str
    probabilities: tuple[float, ...]
    confidence: float
    predictability: float
    uncertainty: float
    disagreement: float
    failure_risk_max: float
    ood_score: float
    update_reason: str
    pit_status: str
    model_weights: tuple[tuple[str, float], ...]


def validate_pit(
    *,
    prediction_time: str,
    available_at: Sequence[str | None],
) -> dict[str, object]:
    pt = _utc(prediction_time)
    blockers = []
    for i, raw in enumerate(available_at):
        if raw is None or str(raw).strip() == "":
            blockers.append(f"row_{i}:available_at_missing")
            continue
        try:
            at = _utc(raw)
        except Exception:
            blockers.append(f"row_{i}:available_at_invalid")
            continue
        if at > pt:
            blockers.append(f"row_{i}:available_at_after_prediction")
    return {"status": "PASS" if not blockers else "FAIL", "blockers": blockers}


def forecast_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = _norm(np.asarray(probabilities))
    if len(y) != len(p):
        raise ValueError("metric input lengths differ")
    pred = np.argmax(p, axis=1)
    k = p.shape[1]
    accuracy = float(np.mean(pred == y))
    ll = float(log_loss(y, p, labels=list(range(k))))
    if k == 2:
        brier = float(brier_score_loss(y, p[:, 1]))
    else:
        brier = float(np.mean(np.sum((p - np.eye(k)[y]) ** 2, axis=1)))
    conf = p.max(axis=1)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(conf[mask].mean()) - float(np.mean((pred == y)[mask]))
            )
    return {"Accuracy": accuracy, "LogLoss": ll, "Brier": brier, "ECE": float(ece)}


def run_e2e_case(
    model_probs: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    *,
    prediction_time: str,
    model_loss_history: Mapping[str, Sequence[float]],
    data_quality: float = 1.0,
    drift_score: float = 0.0,
    ood_score: float = 0.0,
    information_candidates: Sequence[Mapping[str, float]] = (),
    previous_probs: np.ndarray | None = None,
    future_regime_fit: Mapping[str, float] | None = None,
    retrieval_probs: np.ndarray | None = None,
) -> dict[str, object]:
    p_norm = {name: _norm(prob) for name, prob in model_probs.items()}
    if not p_norm:
        raise ValueError("model_probs is empty")
    if len({len(p) for p in p_norm.values()}) != 1:
        raise ValueError("model probability lengths differ")
    mean_p = np.mean(np.stack(list(p_norm.values()), axis=1), axis=1)
    disagreement = model_disagreement(p_norm)
    predictability = predictability_score(
        mean_p,
        history_probs=mean_p,
        data_quality=data_quality,
        drift_score=drift_score,
    )
    failure = FutureFailureEstimator().fit(model_loss_history).predict(model_loss_history)
    risks = {name: row["failure_risk"] for name, row in failure.items()}
    info = choose_information(information_candidates)
    policy = select_prediction_policy(
        predictability=predictability["predictability"],
        ood_score=ood_score,
        failure_risk_max=max(risks.values()) if risks else 0.0,
        information_value=float(info["score"]),
    )
    losses = {
        name: float(np.mean(np.asarray(history, dtype=float)[-12:]))
        for name, history in model_loss_history.items()
    }
    weights = routing_weights(
        losses,
        failure_risk=risks,
        predictability=predictability["predictability"],
        disagreement=float(disagreement["disagreement_score"][-1]),
        future_regime_fit=future_regime_fit,
    )
    final_p = np.sum(
        np.stack([p_norm[name][-1] for name in weights], axis=0)
        * np.asarray(list(weights.values()))[:, None],
        axis=0,
    )
    retrieval_used = False
    if policy == PredictionAction.RETRIEVAL.value and retrieval_probs is not None:
        rp = _norm(np.asarray(retrieval_probs))
        if rp.shape == (1, len(final_p)):
            final_p = 0.80 * final_p + 0.20 * rp[0]
            retrieval_used = True
    final_p = _norm(final_p.reshape(1, -1))[0]
    revision = decide_revision(
        previous_probs,
        final_p.reshape(1, -1),
        information_value=float(info["score"]),
    )
    pit = validate_pit(prediction_time=prediction_time, available_at=[prediction_time])
    uncertainty = float(np.clip(
        0.45 * float(disagreement["normalized_entropy"][-1])
        + 0.30 * (1.0 - predictability["predictability"])
        + 0.15 * float(drift_score)
        + 0.10 * float(ood_score),
        0.0,
        1.0,
    ))
    now = _utc(prediction_time)
    valid_until = now + pd.Timedelta(
        minutes=max(5.0, 60.0 * (0.25 + predictability["predictability"]))
    )
    contract = ForecastContract(
        prediction_time=now.isoformat(),
        valid_until=valid_until.isoformat(),
        model_version="research-v13",
        strategy=policy,
        confidence=predictability["confidence"],
        predictability=predictability["predictability"],
        uncertainty=uncertainty,
        pit_status=str(pit["status"]),
    )
    ledger = LedgerRecord(
        prediction_time=contract.prediction_time,
        strategy=policy,
        action=str(revision["action"]),
        probabilities=tuple(float(x) for x in final_p),
        confidence=contract.confidence,
        predictability=contract.predictability,
        uncertainty=uncertainty,
        disagreement=float(disagreement["disagreement_score"][-1]),
        failure_risk_max=float(max(risks.values()) if risks else 0.0),
        ood_score=float(ood_score),
        update_reason="information_value" if float(info["score"]) > 0 else "state_routing",
        pit_status=contract.pit_status,
        model_weights=tuple((name, float(value)) for name, value in weights.items()),
    )
    return {
        "status": "READY" if pit["status"] == "PASS" else "BLOCKED",
        "disagreement": disagreement,
        "predictability": predictability,
        "future_failure": failure,
        "time_to_failure": {name: row["time_to_failure_periods"] for name, row in failure.items()},
        "information": info,
        "policy": policy,
        "routing": weights,
        "prediction": final_p,
        "revision": revision,
        "retrieval_used": retrieval_used,
        "uncertainty": uncertainty,
        "contract": asdict(contract),
        "ledger": asdict(ledger),
        "diagnostic_metrics": forecast_metrics(np.asarray(y_true), mean_p),
        "audit": {"pit": pit, "ood": float(ood_score), "drift": float(drift_score)},
    }


def _self_test() -> dict[str, object]:
    rng = np.random.default_rng(7)
    n, k = 260, 3
    y = rng.integers(0, k, size=n)
    base = np.eye(k)[y] * 0.66 + 0.17
    models, losses = {}, {}
    for i, noise in enumerate((0.015, 0.035, 0.055)):
        p = _norm(np.clip(base + rng.normal(0.0, noise, size=(n, k)), 0.01, None))
        models[f"M{i}"] = p
        losses[f"M{i}"] = -np.log(np.clip(p[np.arange(n), y], EPS, 1.0))
    result = run_e2e_case(
        models,
        y,
        prediction_time=datetime.now(timezone.utc).isoformat(),
        model_loss_history=losses,
        data_quality=0.94,
        drift_score=0.08,
        ood_score=0.05,
        information_candidates=(
            {"name": "starter_update", "expected_gain": 0.11, "cost": 0.02, "failure_risk": 0.01},
            {"name": "weather_refresh", "expected_gain": 0.04, "cost": 0.03, "failure_risk": 0.03},
        ),
    )
    assert result["status"] == "READY"
    assert abs(sum(result["routing"].values()) - 1.0) < 1e-9
    assert 0.0 <= result["predictability"]["predictability"] <= 1.0
    assert len(result["prediction"]) == k
    return {
        "status": result["status"],
        "policy": result["policy"],
        "routing": result["routing"],
        "predictability": result["predictability"]["predictability"],
        "failure": {name: row["failure_risk"] for name, row in result["future_failure"].items()},
        "time_to_failure": result["time_to_failure"],
        "diagnostic_metrics": result["diagnostic_metrics"],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
