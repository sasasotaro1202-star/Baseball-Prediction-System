"""Research-only extreme feature-representation laboratory.

This layer complements the family-combination laboratory by changing only the
mathematical representation of already PIT-safe features. No new data source
is introduced. All representation choices are fixed before a candidate is
evaluated and the newest holdout is locked until the final winner is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from baseball_backtest import BaseballBacktest
from evaluation.metrics import expected_calibration_error, multiclass_brier
from research.feature_set_variants import feature_family
from research.ultimate_pattern_lab import (
    _family_pattern_catalog,
    _fit_predict,
    _load_games,
    _pattern_id,
    _staged_folds,
    select_pattern,
)

REPRESENTATIONS = (
    "LEVEL",
    "HOME_AWAY_ONLY",
    "GAP_ONLY",
    "GAP_ABS",
    "GAP_POLY2",
    "GAP_SIGNED_LOG",
    "LEVEL_GAP_ABS_LOG",
    "PAIR_LOG_RATIO",
)

SEED_FAMILY_PATTERNS = (
    frozenset(),
    frozenset({"volatility"}),
    frozenset({"starter"}),
    frozenset({"bullpen"}),
    frozenset({"offense"}),
    frozenset({"interaction"}),
    frozenset({"lineup"}),
    frozenset({"weather"}),
    frozenset({"context"}),
    frozenset({"starter", "bullpen"}),
    frozenset({"starter", "offense"}),
    frozenset({"bullpen", "offense"}),
    frozenset({"starter", "interaction"}),
    frozenset({"starter", "bullpen", "offense"}),
    frozenset({"starter", "offense", "interaction"}),
    frozenset({"volatility", "starter", "bullpen", "offense", "interaction"}),
    frozenset({"volatility", "starter", "bullpen", "offense", "interaction", "lineup", "weather", "context"}),
)

CORE_HORIZONS = ("ALL", "SHORT", "LONG")
HALF_LIVES = (600, 900, 1800, 3600, 7200)
MODEL_POOLS = ("LINEAR_TREE", "BROAD_TREE", "DIVERSE")
MODEL_PROFILES = ("BALANCED", "ROBUST", "SMOOTH", "DEEP", "LOCAL", "REGULARIZED")


def _set_model_profile(profile: str) -> None:
    profile = str(profile).upper()
    if profile not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile: {profile}")
    os.environ["BASEBALL_MODEL_PROFILE"] = profile


def _metrics(y, p, league):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    if league == "NPB":
        return {
            "Accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
            "LogLoss": float(log_loss(y, p, labels=[0, 1, 2])),
            "Brier": float(multiclass_brier(y, p, classes=[0, 1, 2])),
            "ECE": float(expected_calibration_error(y, p, classes=[0, 1, 2])),
            "rows": int(len(y)),
        }
    positive = p[:, 0]
    return {
        "Accuracy": float(accuracy_score(y, (positive >= 0.5).astype(int))),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "Brier": float(brier_score_loss((y == 0).astype(int), positive)),
        "ECE": float(expected_calibration_error(y, p, classes=[0, 1])),
        "rows": int(len(y)),
    }


def _safe_pair_bases(columns: list[str]) -> list[str]:
    names = set(map(str, columns))
    return sorted(
        c[2:] for c in names
        if c.startswith("h_") and f"a_{c[2:]}" in names
    )


def _safe_ratio_bases(columns: list[str]) -> list[str]:
    whitelist = (
        "_avg_", "_era", "_whip", "_k9", "_bb9", "_hr9", "_fip",
        "_ip_", "_pitches", "_starts", "_matches", "_gf_", "_ga_",
        "_ab_", "_hr_", "_bb_", "_so_", "_xbh_", "_rest_days",
    )
    bases = _safe_pair_bases(columns)
    return [b for b in bases if any(token in b for token in whitelist)]


def transform_representation(X, mode: str):
    mode = str(mode).upper()
    if mode not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {mode}")
    columns = list(map(str, X.columns))
    stable = [c for c in columns if c in {"home_adv", "expected_env", "context_pit_safe"}]
    hcols = [c for c in columns if c.startswith("h_")]
    acols = [c for c in columns if c.startswith("a_")]
    dcols = [c for c in columns if c.startswith("d_")]
    other = [c for c in columns if c not in set(hcols + acols + dcols + stable)]

    if mode == "LEVEL":
        cols = columns
        out = X.loc[:, cols].copy()
    elif mode == "HOME_AWAY_ONLY":
        cols = stable + hcols + acols + other
        out = X.loc[:, cols].copy()
    elif mode == "GAP_ONLY":
        cols = stable + dcols + other
        out = X.loc[:, cols].copy()
    else:
        cols = columns if mode in {"LEVEL_GAP_ABS_LOG", "PAIR_LOG_RATIO"} else stable + dcols + other
        out = X.loc[:, cols].copy()

    if mode in {"GAP_ABS", "LEVEL_GAP_ABS_LOG"}:
        for c in dcols:
            out[f"abs_{c}"] = np.abs(X[c].to_numpy(dtype=float))
    if mode in {"GAP_POLY2"}:
        for c in dcols:
            v = X[c].to_numpy(dtype=float)
            out[f"sq_{c}"] = v * v
    if mode in {"GAP_SIGNED_LOG", "LEVEL_GAP_ABS_LOG"}:
        for c in dcols:
            v = X[c].to_numpy(dtype=float)
            out[f"slog_{c}"] = np.sign(v) * np.log1p(np.abs(v))
    if mode == "PAIR_LOG_RATIO":
        for base in _safe_ratio_bases(columns):
            h = np.clip(X[f"h_{base}"].to_numpy(dtype=float), 0.0, None)
            a = np.clip(X[f"a_{base}"].to_numpy(dtype=float), 0.0, None)
            out[f"logratio_{base}"] = np.log1p(h) - np.log1p(a)

    out = out.replace([np.inf, -np.inf], np.nan)
    if out.isna().any().any():
        raise RuntimeError(f"representation {mode} generated non-finite values")
    if out.columns.duplicated().any():
        raise RuntimeError(f"representation {mode} generated duplicate columns")
    digest = hashlib.sha256("\n".join(map(str, out.columns)).encode()).hexdigest()
    return out, {
        "representation": mode,
        "feature_count": int(out.shape[1]),
        "feature_schema_hash": digest,
        "base_feature_count": int(len(columns)),
        "derived_feature_count": int(out.shape[1] - len(columns)),
        "pair_base_count": int(len(_safe_pair_bases(columns))),
        "ratio_base_count": int(len(_safe_ratio_bases(columns))),
    }


def run(*, league: str, data_dir: str = "data", holdout_fraction: float = 0.20, mlb_start: int = 2020, mlb_end: int = 2026, top_k_a: int = 12, top_k_b: int = 4):
    if top_k_a < top_k_b or top_k_b < 1:
        raise ValueError("top_k_a must be >= top_k_b >= 1")
    os.environ["BASEBALL_FEATURE_SET_VARIANT"] = "FULL_VALIDATED_ENSEMBLE"
    bt = BaseballBacktest(Path(data_dir))
    games = _load_games(bt, league, mlb_start, mlb_end)
    X, y, meta = bt.build_features(games)
    locked_start = int(len(X) * (1.0 - holdout_fraction))
    folds = _staged_folds(locked_start)

    stage_a = []
    fail_a = []
    catalog = SEED_FAMILY_PATTERNS
    cut_a, stop_a = folds["screen"]
    for i, fams in enumerate(catalog, 1):
        for rep in REPRESENTATIONS:
            candidate = f"{_pattern_id(fams)}|rep={rep}"
            bt._check_time_budget(f"extreme_rep_a:{i}/{len(catalog)}:{rep}")
            try:
                base, fmeta = select_pattern(X, fams, horizon="ALL")
                Xv, rmeta = transform_representation(base, rep)
                metrics, selected, validation_scores = _fit_predict(
                    bt, Xv, y, meta, cut_a, stop_a, league, "LINEAR_TREE",
                    fast_oos=True, half_life=1800,
                )
                stage_a.append({
                    "candidate_id": candidate,
                    "feature_meta": fmeta,
                    "representation_meta": rmeta,
                    "metrics": metrics,
                    "selected_models": selected,
                    "validation_scores": {k: float(v) for k, v in validation_scores.items()},
                    "stage": "A",
                })
            except Exception as exc:
                fail_a.append({
                    "candidate_id": candidate,
                    "stage": "A",
                    "failure_type": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    stage_a = sorted(stage_a, key=lambda r: (r["metrics"]["LogLoss"], r["metrics"]["Brier"], r["candidate_id"]))
    if not stage_a:
        raise RuntimeError("representation Stage A produced no usable candidates")

    stage_b = []
    fail_b = []
    cut_b, stop_b = folds["confirm"]
    for rank_a, arow in enumerate(stage_a[:top_k_a], 1):
        fams = frozenset(arow["feature_meta"]["families"])
        rep = str(arow["representation_meta"]["representation"])
        for half_life in HALF_LIVES:
            for pool in MODEL_POOLS:
                candidate = f"{arow['candidate_id']}|hl={half_life}|pool={pool}"
                bt._check_time_budget(f"extreme_rep_b:{rank_a}:{half_life}:{pool}")
                try:
                    base, fmeta = select_pattern(X, fams, horizon="ALL")
                    Xv, rmeta = transform_representation(base, rep)
                    metrics, selected, validation_scores = _fit_predict(
                        bt, Xv, y, meta, cut_b, stop_b, league, pool,
                        fast_oos=True, half_life=half_life,
                    )
                    stage_b.append({
                        "candidate_id": candidate,
                        "source_stage_a_rank": rank_a,
                        "feature_meta": fmeta,
                        "representation_meta": rmeta,
                        "half_life": int(half_life),
                        "model_pool": pool,
                        "metrics": metrics,
                        "selected_models": selected,
                        "validation_scores": {k: float(v) for k, v in validation_scores.items()},
                        "stage": "B",
                    })
                except Exception as exc:
                    fail_b.append({
                        "candidate_id": candidate,
                        "stage": "B",
                        "failure_type": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    stage_b = sorted(stage_b, key=lambda r: (r["metrics"]["LogLoss"], r["metrics"]["Brier"], r["metrics"]["ECE"], r["candidate_id"]))
    if not stage_b:
        raise RuntimeError("representation Stage B produced no usable candidates")

    stage_c = []
    fail_c = []
    cut_c, stop_c = folds["deep"]
    for rank_b, brow in enumerate(stage_b[:top_k_b], 1):
        fams = frozenset(brow["feature_meta"]["families"])
        rep = str(brow["representation_meta"]["representation"])
        half_life = int(brow["half_life"])
        pool = str(brow["model_pool"])
        for profile in MODEL_PROFILES:
            candidate = f"{brow['candidate_id']}|profile={profile}"
            bt._check_time_budget(f"extreme_rep_c:{rank_b}:{profile}")
            try:
                _set_model_profile(profile)
                base, fmeta = select_pattern(X, fams, horizon="ALL")
                Xv, rmeta = transform_representation(base, rep)
                metrics, selected, validation_scores = _fit_predict(
                    bt, Xv, y, meta, cut_c, stop_c, league, pool,
                    fast_oos=False, half_life=half_life,
                )
                stage_c.append({
                    "candidate_id": candidate,
                    "source_stage_b_rank": rank_b,
                    "feature_meta": fmeta,
                    "representation_meta": rmeta,
                    "half_life": half_life,
                    "model_pool": pool,
                    "model_profile": profile,
                    "metrics": metrics,
                    "selected_models": selected,
                    "validation_scores": {k: float(v) for k, v in validation_scores.items()},
                    "stage": "C",
                })
            except Exception as exc:
                fail_c.append({
                    "candidate_id": candidate,
                    "stage": "C",
                    "failure_type": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    stage_c = sorted(stage_c, key=lambda r: (r["metrics"]["LogLoss"], r["metrics"]["Brier"], r["metrics"]["ECE"], r["candidate_id"]))
    if not stage_c:
        raise RuntimeError("representation Stage C produced no usable candidates")
    winner = stage_c[0]

    fams = frozenset(winner["feature_meta"]["families"])
    rep = str(winner["representation_meta"]["representation"])
    _set_model_profile(str(winner["model_profile"]))
    base, winner_fmeta = select_pattern(X, fams, horizon="ALL")
    Xw, winner_rmeta = transform_representation(base, rep)
    os.environ["BASEBALL_RECENCY_HALF_LIFE_GAMES"] = str(winner["half_life"])
    os.environ["BASEBALL_FAST_MODEL_POOL"] = __import__("research.feature_model_matrix", fromlist=["MODEL_POOLS"]).MODEL_POOLS[winner["model_pool"]]
    os.environ["BASEBALL_FAST_OOS"] = "0"
    fitted, validation_scores, selected = bt.fit_ensemble(
        Xw.iloc[:locked_start], y[:locked_start], league, fast_oos=False,
        context_keys=meta.iloc[:locked_start]["competition_key"] if "competition_key" in meta.columns else None,
    )
    if not fitted:
        raise RuntimeError("representation winner holdout fit produced no ensemble")
    p_holdout = bt.ensemble_proba(
        fitted, Xw.iloc[locked_start:], league,
        context_keys=meta.iloc[locked_start:]["competition_key"] if "competition_key" in meta.columns else None,
    )
    holdout = _metrics(y[locked_start:], p_holdout, league)

    failures = fail_a + fail_b + fail_c
    return {
        "schema_version": "baseball-extreme-representation-lab-v1",
        "status": "RESEARCH_ONLY",
        "research_contract_id": "extreme-representation-v1",
        "decision": "NO_AUTO_ADOPTION",
        "selection_basis": "disjoint chronological Stage A -> B -> C OOS; frozen holdout after winner lock",
        "league": league,
        "git_commit_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "total_games": int(len(X)),
        "development_rows": int(locked_start),
        "locked_holdout_rows": int(len(y) - locked_start),
        "representation_catalog": {
            "modes": list(REPRESENTATIONS),
            "model_profiles": list(MODEL_PROFILES),
            "mode_count": len(REPRESENTATIONS),
            "seed_family_pattern_count": len(SEED_FAMILY_PATTERNS),
            "stage_a_requested": len(SEED_FAMILY_PATTERNS) * len(REPRESENTATIONS),
            "catalog_sha256": hashlib.sha256("\n".join(REPRESENTATIONS).encode()).hexdigest(),
        },
        "folds": {
            name: {"start_row": int(a), "end_row": int(b), "start": str(meta.iloc[a]["datetime"]), "end": str(meta.iloc[b - 1]["datetime"])}
            for name, (a, b) in folds.items()
        },
        "stage_counts": {
            "stage_a_requested": len(SEED_FAMILY_PATTERNS) * len(REPRESENTATIONS),
            "stage_a_successful": len(stage_a),
            "stage_a_failed": len(fail_a),
            "stage_b_requested": int(top_k_a * len(HALF_LIVES) * len(MODEL_POOLS)),
            "stage_b_successful": len(stage_b),
            "stage_b_failed": len(fail_b),
            "stage_c_requested": int(top_k_b * len(MODEL_PROFILES)),
            "stage_c_successful": len(stage_c),
            "stage_c_failed": len(fail_c),
        },
        "stage_a_top": stage_a[:32],
        "stage_b_top": stage_b[:64],
        "stage_c_all": stage_c,
        "failures": failures,
        "winner": winner,
        "locked_holdout": {
            "winner_only": True,
            "metrics": holdout,
            "feature_meta": winner_fmeta,
            "representation_meta": winner_rmeta,
            "selected_models": selected,
            "validation_scores": {k: float(v) for k, v in validation_scores.items()},
            "period": {"start": str(meta.iloc[locked_start]["datetime"]), "end": str(meta.iloc[-1]["datetime"])},
        },
        "research_gate": {
            "holdout_locked_before_selection": True,
            "no_auto_adoption": True,
            "failed_patterns_do_not_pass": True,
            "unexpected_execution_failures": len([f for f in failures if f.get("failure_type") == "FAILED"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["NPB", "MLB"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--mlb-start", type=int, default=2020)
    parser.add_argument("--mlb-end", type=int, default=2026)
    parser.add_argument("--top-k-a", type=int, default=12)
    parser.add_argument("--top-k-b", type=int, default=4)
    args = parser.parse_args()
    payload = run(
        league=args.league, data_dir=args.data_dir,
        holdout_fraction=args.holdout_fraction,
        mlb_start=args.mlb_start, mlb_end=args.mlb_end,
        top_k_a=args.top_k_a, top_k_b=args.top_k_b,
    )
    out = Path(args.data_dir).parent / "results" / f"extreme_representation_lab_{args.league.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "league": payload["league"],
        "git_commit_sha": payload["git_commit_sha"],
        "status": payload["status"],
        "stage_counts": payload["stage_counts"],
        "winner": payload["winner"]["candidate_id"],
        "holdout": payload["locked_holdout"]["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
