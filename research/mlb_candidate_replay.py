"""MLB Development-OOS -> Candidate-Lock -> Independent-Holdout lifecycle.

This is the MLB counterpart to the existing NPB replay. It uses the existing
BaseballBacktest model pool, selects candidates only on chronological
Development OOS, locks the candidate, then evaluates it on a later holdout.
Promotion is fail-closed when score/Low-High evidence is unavailable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.calibration import fit_temperature
from baseball_backtest import BaseballBacktest, low_high_probs, score_candidates
from core.atomic_io import atomic_write_json
from evaluation.metrics import classification_metrics
from research.candidates import CandidateSpec, lock_candidate
from research.validation_pipeline import run_validation_pipeline
from research.adoption_gate import GatePolicy

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class MLBReplayConfig:
    holdout_fraction: float = 0.20
    min_train_rows: int = 180
    min_holdout_rows: int = 200
    block_size: int = 60
    retrain_every: int = 180
    calibration_tolerance: float = 0.005
    recency_half_lives: tuple[int, ...] = (900, 1800, 3600)
    recency_variant_top_k: int = 2


def _proba(bt: BaseballBacktest, model: Any, X):
    return bt.align_proba(model.predict_proba(X), model.classes_, "MLB")


def _fit(bt: BaseballBacktest, name: str, X, y):
    models = bt.models("MLB")
    if name not in models:
        raise ValueError(f"unknown MLB model candidate: {name}")
    model = models[name]
    bt._fit_model(model, X, y, bt._sample_weights(len(X)), "MLB")
    return model


def _development(bt, X, y, start, end, names, block, retrain_every):
    actual, outputs = [], {"ProductionEnsemble": []}
    for name in names:
        outputs[name] = []
    if block <= 0 or retrain_every <= 0:
        raise ValueError("block and retrain_every must be > 0")
    windows = 0
    fitted = None
    candidate_fitted = {}
    last_fit_cut = -10**9
    for cut in range(start, end, block):
        stop = min(end, cut + block)
        # Periodic retraining remains strictly chronological: every reused
        # model was trained only on observations before its first OOS block.
        if fitted is None or cut - last_fit_cut >= retrain_every:
            fitted, _, _ = bt.fit_ensemble(X.iloc[:cut], y[:cut], "MLB", fast_oos=True)
            if not fitted:
                continue
            candidate_fitted = {
                name: _fit(bt, name, X.iloc[:cut], y[:cut])
                for name in names
            }
            last_fit_cut = cut
        windows += 1
        actual.append(y[cut:stop])
        outputs["ProductionEnsemble"].append(bt.ensemble_proba(fitted, X.iloc[cut:stop], "MLB"))
        for name in names:
            outputs[name].append(_proba(bt, candidate_fitted[name], X.iloc[cut:stop]))
    if not actual:
        raise RuntimeError("MLB Development OOS produced no valid windows")
    yy = np.concatenate(actual)
    metrics = {}
    for name, chunks in outputs.items():
        if chunks:
            pp = np.vstack(chunks)
            metrics[name] = classification_metrics(yy[:len(pp)], pp, classes=[0, 1])
    return metrics, windows


def _target_metrics(bt: BaseballBacktest, X_train, games_train, games_holdout, X_holdout, p: np.ndarray, score_fit=None) -> tuple[dict[str, float], dict[str, float]]:
    if score_fit is None:
        score_fit = bt.fit_score_ensemble(X_train, games_train["home_score"].astype(float).to_numpy(), games_train["away_score"].astype(float).to_numpy(), "MLB")
    home_true = games_holdout["home_score"].astype(float).to_numpy()
    away_true = games_holdout["away_score"].astype(float).to_numpy()
    expected_home, expected_away, score_hits, hilo_actual, hilo_prob = [], [], [], [], []
    for i in range(len(games_holdout)):
        lam_h, lam_a = bt.predict_scores(score_fit, X_holdout.iloc[[i]], "MLB")
        split = float(np.clip(p[i, 0] - 0.5, -0.35, 0.35))
        lam_h *= 1.0 + 0.08 * split
        lam_a *= 1.0 - 0.08 * split
        expected_home.append(lam_h); expected_away.append(lam_a)
        choices = {x for x, _ in score_candidates(lam_h, lam_a, 4)}
        high = (home_true[i] + away_true[i]) >= 7
        # "その他" is a tail bucket, not an exact score. Keep Top4HitRate
        # conservative and exact-score based.
        score_hits.append(
            (not high)
            and (f"{int(home_true[i])}-{int(away_true[i])}" in choices)
        )
        _, high_p = low_high_probs(lam_h, lam_a)
        hilo_prob.append(high_p); hilo_actual.append(int(high))
    score_mae = float((np.mean(np.abs(np.asarray(expected_home)-home_true)) + np.mean(np.abs(np.asarray(expected_away)-away_true))) / 2.0)
    ya = np.asarray(hilo_actual, dtype=int); hp = np.clip(np.asarray(hilo_prob), 1e-9, 1-1e-9)
    hilo = {"Accuracy": float(np.mean((hp >= 0.5).astype(int) == ya)), "LogLoss": float(-np.mean(ya*np.log(hp)+(1-ya)*np.log(1-hp))), "Brier": float(np.mean((hp-ya)**2)), "ScoreMAE": score_mae, "Top4HitRate": float(np.mean(score_hits)), "rows": int(len(ya))}
    return {"ScoreMAE": score_mae, "Top4HitRate": float(np.mean(score_hits)), "rows": int(len(ya))}, hilo


def _temperature_scale(p: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    logp = np.log(np.clip(np.asarray(p, dtype=float), 1e-12, 1.0))
    z = logp / float(temperature)
    z -= np.max(z, axis=1, keepdims=True)
    q = np.exp(z)
    return q / np.sum(q, axis=1, keepdims=True)


def _fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    return float(fit_temperature(p, y).temperature)


def _fit_with_half_life(bt: BaseballBacktest, name: str, X, y, half_life: int | None):
    models = bt.models("MLB")
    if name not in models:
        raise ValueError(f"unknown MLB model candidate: {name}")
    previous = os.environ.get("BASEBALL_RECENCY_HALF_LIFE_GAMES")
    try:
        if half_life is not None:
            os.environ["BASEBALL_RECENCY_HALF_LIFE_GAMES"] = str(int(half_life))
        bt._fit_model(models[name], X, y, bt._sample_weights(len(X)), "MLB")
    finally:
        if previous is None:
            os.environ.pop("BASEBALL_RECENCY_HALF_LIFE_GAMES", None)
        else:
            os.environ["BASEBALL_RECENCY_HALF_LIFE_GAMES"] = previous
    return models[name]


def _development_variant(bt, X, y, start, end, name, half_life, block, retrain_every):
    chunks=[]; actual=[]; fitted=None; last=-10**9; windows=0
    for cut in range(start,end,block):
        stop=min(end,cut+block)
        if fitted is None or cut-last>=retrain_every:
            fitted=_fit_with_half_life(bt,name,X.iloc[:cut],y[:cut],half_life)
            last=cut
        actual.append(y[cut:stop])
        chunks.append(_proba(bt,fitted,X.iloc[cut:stop]))
        windows+=1
    if not chunks: raise RuntimeError(f"No OOS windows for {name}@{half_life}")
    yy=np.concatenate(actual); pp=np.vstack(chunks)
    return classification_metrics(yy[:len(pp)],pp,classes=[0,1]),pp,windows


def run_mlb_candidate_cycle(*, data_dir: str | Path = "data", git_commit: str,
                            mlb_start: int = 2020, mlb_end: int = 2026,
                            feature_version: str = "baseball-features-v1",
                            config: MLBReplayConfig = MLBReplayConfig()) -> dict[str, Any]:
    bt = BaseballBacktest(Path(data_dir))
    games = bt.load_mlb(mlb_start, mlb_end)
    if __import__("os").getenv("PIT_SAFE_STARTER_DATA", "0") != "1":
        games = games.copy()
        games["home_starter"] = ""
        games["away_starter"] = ""
        games["confirmed_starters"] = False
        games["starter_evidence_status"] = "not_pit_safe"
    X, y, _meta = bt.build_features(games)
    n = len(X)
    holdout_start = int(n * (1.0 - config.holdout_fraction))
    dev_start = max(config.min_train_rows, int(n * 0.10))
    if holdout_start <= dev_start or n - holdout_start < config.min_holdout_rows:
        raise RuntimeError("MLB replay does not have enough rows for an independent holdout")

    names = list(bt.models("MLB").keys())
    development, windows = _development(bt, X, y, dev_start, holdout_start, names, config.block_size, config.retrain_every)
    baseline = development["ProductionEnsemble"]
    candidates = [(k,v) for k,v in development.items() if k!="ProductionEnsemble"]
    candidates.sort(key=lambda kv:(kv[1]["LogLoss"],kv[1]["Brier"],-kv[1]["Accuracy"],kv[0]))
    if not candidates:
        raise RuntimeError("no MLB candidate produced valid Development OOS metrics")

    base_half_life=int(os.getenv("BASEBALL_RECENCY_HALF_LIFE_GAMES","1800"))
    variant_specs={}
    development_predictions={}
    # Re-run the strict Development OOS once to retain raw predictions for
    # low-complexity calibration and recency challengers.
    actual_dev=[]; fitted_base={}; last_fit=-10**9
    for cut in range(dev_start,holdout_start,config.block_size):
        stop=min(holdout_start,cut+config.block_size)
        if not fitted_base or cut-last_fit>=config.retrain_every:
            for name in ["ProductionEnsemble"] + [k for k,_ in candidates[:max(1,config.recency_variant_top_k)]]:
                if name=="ProductionEnsemble":
                    fitted_base[name]=None
                else:
                    fitted_base[name]=_fit_with_half_life(bt,name,X.iloc[:cut],y[:cut],None)
            last_fit=cut
        actual_dev.append(y[cut:stop])
        if fitted_base:
            for name,model in list(fitted_base.items()):
                if name=="ProductionEnsemble":
                    continue
                development_predictions.setdefault(name,[]).append(_proba(bt,model,X.iloc[cut:stop]))
    y_dev=np.concatenate(actual_dev) if actual_dev else np.empty(0,dtype=int)
    for name in list(development_predictions):
        if development_predictions[name]:
            development_predictions[name]=np.vstack(development_predictions[name])

    for base_name,_m in candidates[:max(1,config.recency_variant_top_k)]:
        for half_life in config.recency_half_lives:
            if int(half_life)==base_half_life:
                continue
            vname=f"{base_name}@recency_hl={int(half_life)}"
            try:
                vm,vp,vw=_development_variant(bt,X,y,dev_start,holdout_start,base_name,int(half_life),config.block_size,config.retrain_every)
            except Exception as exc:
                bt.audit.append({"type":"mlb_recency_variant_error","model":base_name,"recency_half_life":int(half_life),"error":f"{type(exc).__name__}: {exc}"})
                continue
            development[vname]=vm
            development_predictions[vname]=vp
            variant_specs[vname]=(base_name,int(half_life))

    candidates=[(k,v) for k,v in development.items() if k!="ProductionEnsemble"]
    candidates.sort(key=lambda kv:(kv[1]["LogLoss"],kv[1]["Brier"],-kv[1]["Accuracy"],kv[0]))
    calibration_specs={}
    for name,_m in candidates[:max(5,config.recency_variant_top_k+1)]:
        pred=development_predictions.get(name)
        if pred is None or len(pred)!=len(y_dev):
            continue
        t=_fit_temperature(y_dev,pred)
        q=_temperature_scale(pred,t)
        cname=f"TemperatureScaled:{name}"
        development[cname]=classification_metrics(y_dev,q,classes=[0,1])
        development_predictions[cname]=q
        calibration_specs[cname]=(name,t)

    candidates=[(k,v) for k,v in development.items() if k!="ProductionEnsemble"]
    candidates.sort(key=lambda kv:(kv[1]["LogLoss"],kv[1]["Brier"],-kv[1]["Accuracy"],kv[0]))
    selected_name,selected_metrics=candidates[0]
    selected_temperature=1.0
    base_selected=selected_name
    if selected_name.startswith("TemperatureScaled:"):
        base_selected=selected_name[len("TemperatureScaled:"):]
        selected_temperature=calibration_specs[selected_name][1]
    selected_model_name,selected_half_life=variant_specs.get(base_selected,(base_selected,None))

    if baseline["LogLoss"] - selected_metrics["LogLoss"] <= 0:
        return {"stage":"development_evaluated","decision":"NO_CHANGE","baseline":baseline,
                "candidate":selected_metrics,"candidate_model":selected_model_name,
                "candidate_recency_half_life":selected_half_life,"candidate_temperature":selected_temperature,
                "validation_windows":windows,"development":development}

    # Hash the actual ordered dataset content, not only row count/index. This
    # makes candidate identity sensitive to data changes and strengthens
    # reproducibility/auditability.
    dataset_bytes = games.to_json(
        orient="split", date_format="iso", double_precision=15
    ).encode("utf-8")
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    spec = CandidateSpec(
        candidate_id="cand-" + hashlib.sha256(f"MLB|{selected_model_name}|HL={selected_half_life or base_half_life}|T={selected_temperature:.4f}|{feature_version}|{git_commit}|{dataset_hash}".encode()).hexdigest()[:20],
        league="MLB", objective="win", model_version=selected_model_name, feature_version=feature_version,
        development_metrics=selected_metrics,
        selection_reason=f"Development OOS only; selected {selected_model_name} with recency_half_life={selected_half_life or base_half_life} and frozen temperature={selected_temperature:.4f}.",
        git_commit=git_commit, dataset_hash=dataset_hash,
    )
    locked = lock_candidate(spec)

    X_train, X_holdout = X.iloc[:holdout_start], X.iloc[holdout_start:]
    y_train, y_holdout = y[:holdout_start], y[holdout_start:]
    base_fit, _, _ = bt.fit_ensemble(X_train, y_train, "MLB")
    if not base_fit:
        raise RuntimeError("MLB production ensemble could not be fitted for holdout")
    base_p = bt.ensemble_proba(base_fit, X_holdout, "MLB")
    cand_p = _proba(bt, _fit_with_half_life(bt, selected_model_name, X_train, y_train, selected_half_life), X_holdout)
    if selected_temperature != 1.0:
        cand_p = _temperature_scale(cand_p, selected_temperature)
    base = classification_metrics(y_holdout, base_p, classes=[0, 1])
    cand = classification_metrics(y_holdout, cand_p, classes=[0, 1])

    games_train = games.iloc[:holdout_start].reset_index(drop=True)
    games_holdout = games.iloc[holdout_start:].reset_index(drop=True)
    X_train = X.iloc[:holdout_start]
    X_holdout = X.iloc[holdout_start:]
    score_fit = bt.fit_score_ensemble(X_train, games_train["home_score"].astype(float).to_numpy(), games_train["away_score"].astype(float).to_numpy(), "MLB")
    if score_fit is None:
        raise RuntimeError("MLB score model could not be fitted for locked holdout")
    base_score, base_hilo = _target_metrics(bt, X_train, games_train, games_holdout, X_holdout, base_p, score_fit)
    cand_score, cand_hilo = _target_metrics(bt, X_train, games_train, games_holdout, X_holdout, cand_p, score_fit)
    lifecycle = run_validation_pipeline(
        candidate_id=spec.candidate_id,
        development_metrics=selected_metrics,
        holdout_baseline=base,
        holdout_candidate=cand,
        validation_windows=windows,
        calibration_ok=cand["ECE"] <= base["ECE"] + 0.005,
        no_future_target_data=True,
        reproducible=True,
        pit_starter_evidence_ok=False,
        holdout_score_baseline=base_score,
        holdout_score_candidate=cand_score,
        holdout_hilo_baseline=base_hilo,
        holdout_hilo_candidate=cand_hilo,
        league="MLB",
        policy=GatePolicy(require_pit_starter_evidence=True),
    )
    out = {"stage": "locked_holdout_evaluated", "candidate": locked,
           "holdout": {"baseline": base, "candidate": cand,
                       "baseline_score": base_score, "candidate_score": cand_score,
                       "baseline_hilo": base_hilo, "candidate_hilo": cand_hilo},
           "validation": asdict(lifecycle), "decision": lifecycle.decision,
           "score_hilo_status": "CONNECTED_PIT_SAFE_TRAINING_ONLY"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS / "mlb_candidate_development.json", development)
    atomic_write_json(RESULTS / "mlb_locked_holdout.json", out)
    return out
