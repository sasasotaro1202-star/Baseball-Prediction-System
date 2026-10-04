"""Research-only staged baseball pattern laboratory.

Stage A: exhaustive optional-family combinations on an early chronological OOS block.
Stage B: top-K patterns x core-horizon x recency half-life x model-pool on the next block.
Stage C: top patterns with full ensemble/routing/calibration on a later block.
Finally: Development-selected winner only on the locked newest holdout.

No stage is allowed to use the frozen holdout for selection or tuning.
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
from research.feature_model_matrix import MODEL_POOLS

OPTIONAL_FAMILIES = (
    "volatility",
    "starter",
    "bullpen",
    "offense",
    "interaction",
    "lineup",
    "weather",
    "context",
)
CORE_HORIZONS = ("ALL", "SHORT", "LONG")
HALF_LIVES = (600, 900, 1800, 3600, 7200)
STAGE_POOLS = ("LINEAR_TREE", "BROAD_TREE", "DIVERSE")


def _family_pattern_catalog() -> list[frozenset[str]]:
    patterns = []
    for mask in range(1 << len(OPTIONAL_FAMILIES)):
        selected = frozenset(
            family for idx, family in enumerate(OPTIONAL_FAMILIES) if (mask >> idx) & 1
        )
        patterns.append(selected)
    return patterns


def _pattern_id(families: frozenset[str], horizon: str = "ALL") -> str:
    fam = "+".join(sorted(families)) or "NONE"
    return f"CORE|families={fam}|horizon={horizon}"


def _keep_core(name: str, horizon: str) -> bool:
    if horizon == "ALL":
        return True
    fam = feature_family(name)
    if fam != "core":
        return True
    stable = {"home_adv", "expected_env"}
    if name in stable:
        return True
    if horizon == "SHORT":
        return name.endswith(("_3", "_5", "_10"))
    if horizon == "LONG":
        return name.endswith(("_10", "_20", "_30", "_45", "_60"))
    raise ValueError(f"unknown core horizon: {horizon}")


def select_pattern(X, families: frozenset[str], *, horizon: str = "ALL"):
    available = {feature_family(str(c)) for c in X.columns}
    required = set(families) - {"core"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"pattern requires unavailable feature families: {','.join(missing)}"
        )
    allowed = set(families) | {"core"}
    cols = [
        str(c)
        for c in X.columns
        if feature_family(str(c)) in allowed and _keep_core(str(c), horizon)
    ]
    if not cols:
        raise ValueError(f"pattern selected zero columns: {_pattern_id(families, horizon)}")
    out = X.loc[:, cols].copy()
    digest = hashlib.sha256("\n".join(cols).encode()).hexdigest()
    return out, {
        "pattern_id": _pattern_id(families, horizon),
        "families": sorted(families),
        "core_horizon": horizon,
        "feature_count": int(len(cols)),
        "feature_schema_hash": digest,
        "feature_family_counts": {
            fam: int(sum(feature_family(c) == fam for c in cols))
            for fam in sorted({feature_family(c) for c in cols})
        },
    }


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


def _load_games(bt, league, mlb_start, mlb_end):
    if league == "NPB":
        games = bt.aggregate_npb_games(bt.load_npb_pbp())
    else:
        games = bt.load_mlb(mlb_start, mlb_end)
    if "game_class" in games:
        games = games[games["game_class"].astype(str).eq("official")]
    if "season_type" in games:
        games = games[games["season_type"].astype(str).eq("regular_season")]
    games = games.sort_values(["datetime", "game_id"]).reset_index(drop=True)
    if len(games) < 500:
        raise RuntimeError(f"{league}: insufficient games={len(games)}")
    return games


def _staged_folds(locked_start: int) -> dict[str, tuple[int, int]]:
    ratios = {
        "screen": (0.50, 0.60),
        "confirm": (0.60, 0.70),
        "deep": (0.70, 0.80),
    }
    out = {}
    for name, (a, b) in ratios.items():
        start = int(locked_start * a)
        stop = int(locked_start * b)
        if start < 120 or stop <= start + 40:
            raise RuntimeError(f"development band too short: {name} {start}:{stop}")
        out[name] = (start, stop)
    return out


def _fit_predict(bt, Xv, y, meta, cut, stop, league, pool, *, fast_oos: bool, half_life: int):
    os.environ["BASEBALL_RECENCY_HALF_LIFE_GAMES"] = str(half_life)
    os.environ["BASEBALL_FAST_MODEL_POOL"] = MODEL_POOLS[pool]
    os.environ["BASEBALL_FAST_OOS"] = "1" if fast_oos else "0"
    fitted, validation_scores, selected = bt.fit_ensemble(
        Xv.iloc[:cut],
        y[:cut],
        league,
        fast_oos=fast_oos,
        context_keys=meta.iloc[:cut]["competition_key"]
        if "competition_key" in meta.columns else None,
    )
    if not fitted:
        raise RuntimeError(f"no fitted ensemble at cut={cut}")
    p = bt.ensemble_proba(
        fitted,
        Xv.iloc[cut:stop],
        league,
        context_keys=meta.iloc[cut:stop]["competition_key"]
        if "competition_key" in meta.columns else None,
    )
    metrics = _metrics(y[cut:stop], p, league)
    return metrics, selected, validation_scores


def run(
    *,
    league: str,
    data_dir: str = "data",
    holdout_fraction: float = 0.20,
    mlb_start: int = 2020,
    mlb_end: int = 2026,
    top_k_b: int = 8,
    top_k_c: int = 4,
) -> dict[str, Any]:
    if not 0.15 <= holdout_fraction <= 0.30:
        raise ValueError("holdout_fraction must be 0.15..0.30")
    if top_k_b < top_k_c or top_k_c < 1:
        raise ValueError("top_k_b must be >= top_k_c >= 1")

    os.environ["BASEBALL_FEATURE_SET_VARIANT"] = "FULL_VALIDATED_ENSEMBLE"
    bt = BaseballBacktest(Path(data_dir))
    games = _load_games(bt, league, mlb_start, mlb_end)
    X, y, meta = bt.build_features(games)
    locked_start = int(len(X) * (1.0 - holdout_fraction))
    folds = _staged_folds(locked_start)
    catalog = _family_pattern_catalog()
    available_families = {feature_family(str(c)) for c in X.columns}

    screen_rows: list[dict[str, Any]] = []
    fail_a: list[dict[str, Any]] = []
    cut_a, stop_a = folds["screen"]
    for idx, fams in enumerate(catalog, 1):
        bt._check_time_budget(f"ultimate_stage_a:{idx}/{len(catalog)}")
        candidate = _pattern_id(fams, "ALL")
        try:
            Xv, fmeta = select_pattern(X, fams, horizon="ALL")
            metrics, selected, validation_scores = _fit_predict(
                bt, Xv, y, meta, cut_a, stop_a, league, "LINEAR_TREE",
                fast_oos=True, half_life=1800,
            )
            screen_rows.append({
                "candidate_id": candidate,
                "feature_meta": fmeta,
                "metrics": metrics,
                "selected_models": selected,
                "validation_scores": {k: float(v) for k, v in validation_scores.items()},
                "stage": "A",
            })
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failure_type = "BLOCKED_UNAVAILABLE_FAMILY" if "requires unavailable feature families" in msg else "FAILED"
            fail_a.append({
                "candidate_id": candidate,
                "stage": "A",
                "failure_type": failure_type,
                "error": msg,
            })
    stage_a = sorted(
        screen_rows,
        key=lambda r: (r["metrics"]["LogLoss"], r["metrics"]["Brier"], r["candidate_id"]),
    )
    if not stage_a:
        raise RuntimeError("Stage A produced no usable candidates")
    top_a = stage_a[:top_k_b]

    stage_b: list[dict[str, Any]] = []
    fail_b: list[dict[str, Any]] = []
    cut_b, stop_b = folds["confirm"]
    for rank_a, arow in enumerate(top_a, 1):
        fams = frozenset(arow["feature_meta"]["families"])
        for horizon in CORE_HORIZONS:
            for half_life in HALF_LIVES:
                for pool in STAGE_POOLS:
                    candidate = f"{arow['candidate_id']}|horizon={horizon}|hl={half_life}|pool={pool}"
                    bt._check_time_budget(f"ultimate_stage_b:{candidate}")
                    try:
                        Xv, fmeta = select_pattern(X, fams, horizon=horizon)
                        metrics, selected, validation_scores = _fit_predict(
                            bt, Xv, y, meta, cut_b, stop_b, league, pool,
                            fast_oos=True, half_life=half_life,
                        )
                        stage_b.append({
                            "candidate_id": candidate,
                            "source_stage_a_rank": rank_a,
                            "feature_meta": fmeta,
                            "half_life": half_life,
                            "model_pool": pool,
                            "metrics": metrics,
                            "selected_models": selected,
                            "validation_scores": {k: float(v) for k, v in validation_scores.items()},
                            "stage": "B",
                        })
                    except Exception as exc:
                        msg = f"{type(exc).__name__}: {exc}"
                        failure_type = "BLOCKED_UNAVAILABLE_FAMILY" if "requires unavailable feature families" in msg else "FAILED"
                        fail_b.append({
                            "candidate_id": candidate,
                            "stage": "B",
                            "failure_type": failure_type,
                            "error": msg,
                        })
    stage_b = sorted(
        stage_b,
        key=lambda r: (r["metrics"]["LogLoss"], r["metrics"]["Brier"], r["metrics"]["ECE"], r["candidate_id"]),
    )
    if not stage_b:
        raise RuntimeError("Stage B produced no usable candidates")
    top_b = stage_b[:top_k_c]

    stage_c: list[dict[str, Any]] = []
    fail_c: list[dict[str, Any]] = []
    cut_c, stop_c = folds["deep"]
    for rank_b, brow in enumerate(top_b, 1):
        fams = frozenset(brow["feature_meta"]["families"])
        horizon = str(brow["feature_meta"]["core_horizon"])
        half_life = int(brow["half_life"])
        pool = str(brow["model_pool"])
        candidate = brow["candidate_id"]
        bt._check_time_budget(f"ultimate_stage_c:{candidate}")
        try:
            Xv, fmeta = select_pattern(X, fams, horizon=horizon)
            metrics, selected, validation_scores = _fit_predict(
                bt, Xv, y, meta, cut_c, stop_c, league, pool,
                fast_oos=False, half_life=half_life,
            )
            stage_c.append({
                "candidate_id": candidate,
                "source_stage_b_rank": rank_b,
                "feature_meta": fmeta,
                "half_life": half_life,
                "model_pool": pool,
                "metrics": metrics,
                "selected_models": selected,
                "validation_scores": {k: float(v) for k, v in validation_scores.items()},
                "stage": "C",
            })
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failure_type = "BLOCKED_UNAVAILABLE_FAMILY" if "requires unavailable feature families" in msg else "FAILED"
            fail_c.append({
                "candidate_id": candidate,
                "stage": "C",
                "failure_type": failure_type,
                "error": msg,
            })
    stage_c = sorted(
        stage_c,
        key=lambda r: (r["metrics"]["LogLoss"], r["metrics"]["Brier"], r["metrics"]["ECE"], r["candidate_id"]),
    )
    if not stage_c:
        raise RuntimeError("Stage C produced no usable candidates")
    winner = stage_c[0]

    fams = frozenset(winner["feature_meta"]["families"])
    Xw, winner_fmeta = select_pattern(X, fams, horizon=str(winner["feature_meta"]["core_horizon"]))
    os.environ["BASEBALL_RECENCY_HALF_LIFE_GAMES"] = str(int(winner["half_life"]))
    os.environ["BASEBALL_FAST_MODEL_POOL"] = MODEL_POOLS[str(winner["model_pool"])]
    os.environ["BASEBALL_FAST_OOS"] = "0"
    fitted, validation_scores, selected = bt.fit_ensemble(
        Xw.iloc[:locked_start],
        y[:locked_start],
        league,
        fast_oos=False,
        context_keys=meta.iloc[:locked_start]["competition_key"]
        if "competition_key" in meta.columns else None,
    )
    if not fitted:
        raise RuntimeError("winner holdout fit produced no ensemble")
    p_holdout = bt.ensemble_proba(
        fitted,
        Xw.iloc[locked_start:],
        league,
        context_keys=meta.iloc[locked_start:]["competition_key"]
        if "competition_key" in meta.columns else None,
    )
    holdout = _metrics(y[locked_start:], p_holdout, league)

    all_failures = fail_a + fail_b + fail_c
    return {
        "schema_version": "baseball-ultimate-pattern-lab-v1",
        "status": "RESEARCH_ONLY",
        "decision": "NO_AUTO_ADOPTION",
        "selection_basis": "disjoint chronological Stage A -> B -> C OOS; frozen holdout after winner lock",
        "league": league,
        "total_games": int(len(X)),
        "development_rows": int(locked_start),
        "locked_holdout_rows": int(len(y) - locked_start),
        "feature_family_catalog": {
            "optional_families": list(OPTIONAL_FAMILIES),
            "catalog_size": int(len(catalog)),
            "available_families": sorted(available_families),
            "catalog_sha256": hashlib.sha256(
                "\n".join(_pattern_id(x) for x in catalog).encode()
            ).hexdigest(),
        },
        "folds": {
            name: {
                "start_row": int(a),
                "end_row": int(b),
                "start": str(meta.iloc[a]["datetime"]),
                "end": str(meta.iloc[b - 1]["datetime"]),
            }
            for name, (a, b) in folds.items()
        },
        "stage_counts": {
            "stage_a_requested": int(len(catalog)),
            "stage_a_successful": int(len(stage_a)),
            "stage_a_failed": int(len(fail_a)),
            "stage_b_requested": int(top_k_b * len(CORE_HORIZONS) * len(HALF_LIVES) * len(STAGE_POOLS)),
            "stage_b_successful": int(len(stage_b)),
            "stage_b_failed": int(len(fail_b)),
            "stage_c_requested": int(top_k_c),
            "stage_c_successful": int(len(stage_c)),
            "stage_c_failed": int(len(fail_c)),
        },
        "stage_a_top": stage_a[:min(32, len(stage_a))],
        "stage_b_top": stage_b[:min(64, len(stage_b))],
        "stage_c_all": stage_c,
        "failures": all_failures,
        "winner": winner,
        "locked_holdout": {
            "winner_only": True,
            "metrics": holdout,
            "feature_meta": winner_fmeta,
            "selected_models": selected,
            "validation_scores": {k: float(v) for k, v in validation_scores.items()},
            "period": {
                "start": str(meta.iloc[locked_start]["datetime"]),
                "end": str(meta.iloc[-1]["datetime"]),
            },
        },
        "research_gate": {
            "pit_source": "single FULL_VALIDATED_ENSEMBLE feature build; downstream candidates only drop columns",
            "holdout_locked_before_selection": True,
            "no_auto_adoption": True,
            "failed_patterns_do_not_pass": True,
            "unexpected_execution_failures": int(
                sum(f.get("failure_type") == "FAILED" for f in all_failures)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["NPB", "MLB"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--mlb-start", type=int, default=2020)
    parser.add_argument("--mlb-end", type=int, default=2026)
    parser.add_argument("--top-k-b", type=int, default=8)
    parser.add_argument("--top-k-c", type=int, default=4)
    args = parser.parse_args()
    payload = run(
        league=args.league,
        data_dir=args.data_dir,
        holdout_fraction=args.holdout_fraction,
        mlb_start=args.mlb_start,
        mlb_end=args.mlb_end,
        top_k_b=args.top_k_b,
        top_k_c=args.top_k_c,
    )
    out = Path(args.data_dir).parent / "results" / f"ultimate_pattern_lab_{args.league.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "league": payload["league"],
        "status": payload["status"],
        "stage_counts": payload["stage_counts"],
        "winner": payload["winner"]["candidate_id"],
        "holdout": payload["locked_holdout"]["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
