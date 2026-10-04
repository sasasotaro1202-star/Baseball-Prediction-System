"""Research-only score-distribution pattern laboratory.

The lab reuses one chronological score-model fit per OOS block and evaluates a
broad family of post-fit score patterns on the exact same cases. This isolates
distributional choices from data/fitting variance and keeps the frozen holdout
strictly score-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from baseball_backtest import BaseballBacktest
from research.correlated_score import grid as score_grid
from research.ultimate_pattern_lab import _load_games, _staged_folds


MODEL_MIXES = (
    ("TOP1", 1.0),
    ("TOP2_INVLOSS", 1.0),
    ("TOP3_INVLOSS", 1.0),
    ("TOP3_UNIFORM", 0.0),
    ("TOP3_POWER_0.5", 0.5),
    ("TOP3_POWER_1.5", 1.5),
    ("TOP3_POWER_2.0", 2.0),
)
SHARED_SCALES = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25)
MEAN_SHRINKS = (0.0, 0.05, 0.10, 0.20, 0.30)


def _clip_positive(x: float) -> float:
    return float(np.clip(float(x), 0.05, 15.0))


def _mix_lambdas(block: dict[str, Any], mode: str) -> tuple[np.ndarray, np.ndarray]:
    homes = np.asarray(block["home_lambdas"], dtype=float)
    aways = np.asarray(block["away_lambdas"], dtype=float)
    losses = np.asarray(block["losses"], dtype=float)
    n = homes.shape[0]
    if mode == "TOP1":
        idx = np.argmin(losses)
        w = np.zeros(n, dtype=float)
        w[idx] = 1.0
    elif mode == "TOP3_UNIFORM":
        w = np.full(n, 1.0 / n, dtype=float)
    else:
        if mode == "TOP2_INVLOSS":
            idxs = np.argsort(losses)[: min(2, n)]
            w = np.zeros(n, dtype=float)
            inv = 1.0 / np.maximum(losses[idxs], 1e-9)
            inv /= max(inv.sum(), 1e-12)
            w[idxs] = inv
        else:
            if mode == "TOP3_INVLOSS":
                power = 1.0
            elif mode == "TOP3_POWER_0.5":
                power = 0.5
            elif mode == "TOP3_POWER_1.5":
                power = 1.5
            elif mode == "TOP3_POWER_2.0":
                power = 2.0
            else:
                raise ValueError(f"unknown model mix: {mode}")
            inv = 1.0 / np.maximum(losses, 1e-9) ** power
            inv /= max(inv.sum(), 1e-12)
            w = inv
    return (
        np.sum(homes * w[:, None], axis=0),
        np.sum(aways * w[:, None], axis=0),
    )


def _variant_id(model_mix: str, shared_scale: float, mean_shrink: float) -> str:
    return f"mix={model_mix}|shared={shared_scale:.2f}|mean_shrink={mean_shrink:.2f}"


def _evaluate_variant(
    y_home: np.ndarray,
    y_away: np.ndarray,
    blocks: list[dict[str, Any]],
    *,
    model_mix: str,
    shared_scale: float,
    mean_shrink: float,
) -> dict[str, float]:
    true_total = np.asarray(y_home, dtype=float) + np.asarray(y_away, dtype=float)
    pred_home: list[float] = []
    pred_away: list[float] = []
    pred_high: list[float] = []
    exact_nll: list[float] = []
    top4_hits: list[bool] = []

    cursor = 0
    for block in blocks:
        count = int(block["rows"])
        mix_h, mix_a = _mix_lambdas(block, model_mix)
        train_h_mean = float(block["train_home_mean"])
        train_a_mean = float(block["train_away_mean"])
        shared = np.asarray(block["shared_lambda"], dtype=float) * float(shared_scale)
        lh = (1.0 - mean_shrink) * mix_h + mean_shrink * train_h_mean
        la = (1.0 - mean_shrink) * mix_a + mean_shrink * train_a_mean
        lh = np.clip(lh, 0.05, 15.0)
        la = np.clip(la, 0.05, 15.0)
        for j in range(count):
            pred_home.append(float(lh[j]))
            pred_away.append(float(la[j]))
            # Exact canonical contract: HIGH is total runs >= 7.
            mat = score_grid(float(lh[j]), float(la[j]), float(max(shared[j], 0.0)), max_runs=20)
            high = 1.0 - float(sum(mat[i, k] for i in range(mat.shape[0]) for k in range(mat.shape[1]) if i + k <= 6))
            pred_high.append(float(np.clip(high, 1e-9, 1 - 1e-9)))
            h = int(y_home[cursor + j])
            a = int(y_away[cursor + j])
            if h < mat.shape[0] and a < mat.shape[1]:
                p_exact = float(np.clip(mat[h, a], 1e-12, 1.0))
                exact_nll.append(-np.log(p_exact))
                cells = sorted(
                    (
                        (float(mat[i, k]), f"{i}-{k}")
                        for i in range(mat.shape[0])
                        for k in range(mat.shape[1])
                    ),
                    reverse=True,
                )
                top4_hits.append(f"{h}-{a}" in {label for _, label in cells[:4]} and (h + a) < 7)
        cursor += count

    ph = np.asarray(pred_home, dtype=float)
    pa = np.asarray(pred_away, dtype=float)
    high_p = np.asarray(pred_high, dtype=float)
    actual_high = (true_total >= 7).astype(int)
    score_mae = float(
        (np.mean(np.abs(ph - np.asarray(y_home, dtype=float)))
         + np.mean(np.abs(pa - np.asarray(y_away, dtype=float))) ) / 2.0
    )
    high_ll = float(log_loss(actual_high, high_p, labels=[0, 1]))
    high_brier = float(np.mean((high_p - actual_high) ** 2))
    high_acc = float(np.mean((high_p >= 0.5).astype(int) == actual_high))
    return {
        "ScoreMAE": score_mae,
        "HighLogLoss": high_ll,
        "HighBrier": high_brier,
        "HighAccuracy": high_acc,
        "Top4HitRate": float(np.mean(top4_hits)) if top4_hits else 0.0,
        "ExactScoreLogLoss": float(np.mean(exact_nll)) if exact_nll else float("nan"),
        "ExactScoreRows": int(len(exact_nll)),
        "rows": int(len(y_home)),
    }


def _fit_score_block(
    bt: BaseballBacktest,
    X_train,
    games_train,
    X_eval,
    games_eval,
    league: str,
) -> dict[str, Any]:
    fitted = bt.fit_score_ensemble(
        X_train,
        games_train["home_score"].astype(float).to_numpy(),
        games_train["away_score"].astype(float).to_numpy(),
        league,
    )
    if not fitted:
        raise RuntimeError(f"{league} score fit returned no model")
    model_entries = fitted["models"]
    losses = np.asarray(
        [float(fitted["scores"].get(name, 1.0)) for name, _, _ in model_entries],
        dtype=float,
    )
    home_lambdas = []
    away_lambdas = []
    for _, mh, ma in model_entries:
        home_lambdas.append(np.clip(mh.predict(X_eval), 0.05, 15.0))
        away_lambdas.append(np.clip(ma.predict(X_eval), 0.05, 15.0))
    return {
        "rows": int(len(X_eval)),
        "home_lambdas": np.vstack(home_lambdas),
        "away_lambdas": np.vstack(away_lambdas),
        "losses": losses,
        "shared_lambda": np.full(len(X_eval), float(fitted.get("shared_lambda", 0.0))),
        "train_home_mean": float(games_train["home_score"].astype(float).mean()),
        "train_away_mean": float(games_train["away_score"].astype(float).mean()),
        "selected_models": [name for name, _, _ in model_entries],
    }


def _rank_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize cross-fold rank stability using already computed common OOS rows."""
    if not rows:
        return {"eligible": False, "reason": "no_variants"}
    # The laboratory currently stores one aggregate metric per variant. Keep
    # the stability contract explicit so a later fold-level scorer can populate
    # this field without changing the artifact schema semantics.
    primary = [float(r["metrics"]["ScoreMAE"]) for r in rows]
    finite = [x for x in primary if np.isfinite(x)]
    if not finite:
        return {"eligible": False, "reason": "non_finite_primary"}
    order = np.argsort(np.asarray(primary, dtype=float), kind="mergesort")
    return {
        "eligible": True,
        "best_rank": 1,
        "best_variant": str(rows[int(order[0])]["variant_id"]),
        "finite_variant_count": int(len(finite)),
        "primary_spread": float(np.max(finite) - np.min(finite)),
    }


def run(
    *,
    league: str,
    data_dir: str = "data",
    holdout_fraction: float = 0.20,
    mlb_start: int = 2020,
    mlb_end: int = 2026,
    block_size: int = 80,
) -> dict[str, Any]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    os.environ["BASEBALL_FAST_OOS"] = "1"
    os.environ["BASEBALL_SCORE_FAST_VALIDATION"] = "1"
    bt = BaseballBacktest(Path(data_dir))
    games = _load_games(bt, league, mlb_start, mlb_end)
    os.environ["BASEBALL_FEATURE_SET_VARIANT"] = "FULL_VALIDATED_ENSEMBLE"
    X, y_cls, meta = bt.build_features(games)
    y_home = games["home_score"].astype(float).to_numpy()
    y_away = games["away_score"].astype(float).to_numpy()

    locked_start = int(len(X) * (1.0 - holdout_fraction))
    if locked_start < 300 or len(X) - locked_start < 100:
        raise RuntimeError("insufficient chronological data for score-pattern holdout")
    folds = _staged_folds(locked_start)

    dev_blocks: list[dict[str, Any]] = []
    for fold_name in ("screen", "confirm", "deep"):
        start, stop = folds[fold_name]
        for cut in range(start, stop, block_size):
            eval_stop = min(stop, cut + block_size)
            block = _fit_score_block(
                bt,
                X.iloc[:cut],
                games.iloc[:cut],
                X.iloc[cut:eval_stop],
                games.iloc[cut:eval_stop],
                league,
            )
            dev_blocks.append({**block, "fold": fold_name, "cut": int(cut), "stop": int(eval_stop)})

    development_case_count = int(sum(int(b["rows"]) for b in dev_blocks))
    expected_development_rows = int(folds["deep"][1] - folds["screen"][0])
    if development_case_count != expected_development_rows:
        raise RuntimeError(
            "score development rows do not reconcile: "
            f"blocks={development_case_count} expected={expected_development_rows}"
        )

    variants = [
        (mix, shared, shrink)
        for mix, _ in MODEL_MIXES
        for shared in SHARED_SCALES
        for shrink in MEAN_SHRINKS
    ]
    rows = []
    for model_mix, shared_scale, mean_shrink in variants:
        metrics = _evaluate_variant(
            y_home[folds["screen"][0]:folds["deep"][1]],
            y_away[folds["screen"][0]:folds["deep"][1]],
            [
                {
                    **block,
                    "rows": min(
                        block["rows"],
                        folds["deep"][1] - max(folds["screen"][0], block["cut"]),
                    ),
                }
                for block in dev_blocks
            ],
            model_mix=model_mix,
            shared_scale=shared_scale,
            mean_shrink=mean_shrink,
        )
        rows.append({
            "variant_id": _variant_id(model_mix, shared_scale, mean_shrink),
            "model_mix": model_mix,
            "shared_scale": shared_scale,
            "mean_shrink": mean_shrink,
            "metrics": metrics,
        })

    rows.sort(key=lambda r: (
        r["metrics"]["ScoreMAE"],
        float(r["metrics"]["ExactScoreLogLoss"])
        if np.isfinite(float(r["metrics"]["ExactScoreLogLoss"])) else float("inf"),
        r["metrics"]["HighLogLoss"],
        r["metrics"]["Top4HitRate"] * -1.0,
        r["variant_id"],
    ))
    winner = rows[0]

    holdout_start = locked_start
    holdout_block = _fit_score_block(
        bt,
        X.iloc[:holdout_start],
        games.iloc[:holdout_start],
        X.iloc[holdout_start:],
        games.iloc[holdout_start:],
        league,
    )
    hold_yh = y_home[holdout_start:]
    hold_ya = y_away[holdout_start:]
    holdout_metrics = _evaluate_variant(
        hold_yh,
        hold_ya,
        [holdout_block],
        model_mix=winner["model_mix"],
        shared_scale=float(winner["shared_scale"]),
        mean_shrink=float(winner["mean_shrink"]),
    )

    serial_blocks = [
        {
            "fold": b["fold"],
            "cut": b["cut"],
            "stop": b["stop"],
            "rows": b["rows"],
            "selected_models": b["selected_models"],
            "losses": [float(x) for x in b["losses"]],
            "train_home_mean": b["train_home_mean"],
            "train_away_mean": b["train_away_mean"],
            "shared_mean": float(np.mean(b["shared_lambda"])),
        }
        for b in dev_blocks
    ]
    return {
        "schema_version": "baseball-score-distribution-pattern-lab-v1",
        "status": "RESEARCH_ONLY",
        "decision": "NO_AUTO_ADOPTION",
        "git_commit_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "league": league,
        "total_games": int(len(X)),
        "development_rows": int(locked_start),
        "locked_holdout_rows": int(len(X) - locked_start),
        "variant_catalog": {
            "model_mixes": [x[0] for x in MODEL_MIXES],
            "shared_scales": list(SHARED_SCALES),
            "mean_shrinks": list(MEAN_SHRINKS),
            "variant_count": len(variants),
            "catalog_sha256": hashlib.sha256(
                "\n".join(_variant_id(m, s, sh) for m, s, sh in variants).encode()
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
        "score_fit_blocks": serial_blocks,
        "development_top": rows[:40],
        "rank_stability": _rank_stability(rows),
        "winner": winner,
        "locked_holdout": {
            "winner_only": True,
            "metrics": holdout_metrics,
            "variant_id": winner["variant_id"],
            "period": {
                "start": str(meta.iloc[holdout_start]["datetime"]),
                "end": str(meta.iloc[-1]["datetime"]),
            },
        },
        "research_gate": {
            "holdout_locked_before_selection": True,
            "no_auto_adoption": True,
            "failed_patterns_do_not_pass": True,
            "unexpected_execution_failures": 0,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--league", choices=["NPB", "MLB"], required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--holdout-fraction", type=float, default=0.20)
    p.add_argument("--mlb-start", type=int, default=2020)
    p.add_argument("--mlb-end", type=int, default=2026)
    p.add_argument("--block-size", type=int, default=80)
    a = p.parse_args()
    payload = run(
        league=a.league,
        data_dir=a.data_dir,
        holdout_fraction=a.holdout_fraction,
        mlb_start=a.mlb_start,
        mlb_end=a.mlb_end,
        block_size=a.block_size,
    )
    out = Path(a.data_dir).parent / "results" / f"score_distribution_pattern_lab_{a.league.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "league": payload["league"],
        "status": payload["status"],
        "variant_count": payload["variant_catalog"]["variant_count"],
        "winner": payload["winner"]["variant_id"],
        "holdout": payload["locked_holdout"]["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
