"""Locked-holdout gate for the PIT-safe experience learning policy.

The learner itself is allowed to adapt from matured outcomes. This evaluator
keeps a later chronological holdout completely outside policy construction and
reports whether the learned calibration policy is a promotion candidate.

It never promotes or changes production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.uncertainty import paired_block_bootstrap, to_dict as uncertainty_to_dict
from research.experience_learning import (
    OUTCOME_INDEX,
    apply_policy,
    build_policy,
    load_experience,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_INPUT = ROOT / "data" / "experience" / "experience_ledger.csv"


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(p) != len(y) or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("invalid metric inputs")
    pred = p.argmax(axis=1)
    one = np.zeros_like(p)
    one[np.arange(len(y)), y] = 1.0
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
    return {
        "rows": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "logloss": float(np.mean(ll)),
        "brier": float(np.mean(np.sum((p - one) ** 2, axis=1))),
    }


def _unit_probability(row: pd.Series) -> np.ndarray:
    raw = np.asarray(
        [
            float(row["home_win_pct"]),
            float(row["draw_pct"]),
            float(row["away_win_pct"]),
        ],
        dtype=float,
    )
    total = float(raw.sum())
    if abs(total - 100.0) <= 0.0003:
        raw /= 100.0
    elif abs(total - 1.0) <= 0.000003:
        pass
    else:
        raise ValueError("experience probability row is not normalized")
    if (raw < 0).any() or not np.isfinite(raw).all():
        raise ValueError("experience probability contains invalid values")
    return raw / max(raw.sum(), 1e-12)


def evaluate_locked_holdout(
    frame: pd.DataFrame,
    *,
    holdout_fraction: float = 0.20,
    min_train_cases: int = 30,
    min_holdout_cases: int = 30,
    bootstrap_min_cases: int = 60,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "status": "NO_EXPERIENCE",
            "promotion_status": "HOLD",
            "reason": "no experience cases",
        }
    if not 0.10 <= holdout_fraction <= 0.50:
        raise ValueError("holdout_fraction must be between 0.10 and 0.50")

    work = frame.sort_values(
        ["prediction_cutoff_utc", "prediction_id"], kind="mergesort"
    ).reset_index(drop=True)
    n = len(work)
    split = int(n * (1.0 - holdout_fraction))
    train = work.iloc[:split].copy()
    holdout = work.iloc[split:].copy()

    if len(train) < min_train_cases or len(holdout) < min_holdout_cases:
        return {
            "status": "INSUFFICIENT_CASES",
            "promotion_status": "HOLD",
            "reason": "insufficient chronological cases for locked holdout",
            "rows": int(n),
            "train_cases": int(len(train)),
            "holdout_cases": int(len(holdout)),
            "min_train_cases": int(min_train_cases),
            "min_holdout_cases": int(min_holdout_cases),
        }

    # Policy cutoff is exactly the first holdout prediction cutoff. Outcomes
    # available after this boundary cannot enter the learned policy.
    holdout_cutoff = pd.Timestamp(holdout.iloc[0]["prediction_cutoff_utc"])
    policy = build_policy(
        train,
        cutoff=holdout_cutoff,
        min_group_rows=min_train_cases,
    )

    if policy.get("status") != "READY":
        return {
            "status": "INSUFFICIENT_POLICY",
            "promotion_status": "HOLD",
            "reason": f"policy status={policy.get('status')}",
            "rows": int(n),
            "train_cases": int(len(train)),
            "holdout_cases": int(len(holdout)),
            "policy": {
                "status": policy.get("status"),
                "matured_rows": policy.get("matured_rows"),
            },
        }

    baseline_rows: list[np.ndarray] = []
    learned_rows: list[np.ndarray] = []
    y: list[int] = []
    policy_sources: list[str | None] = []

    for _, row in holdout.iterrows():
        base = _unit_probability(row)
        learned, source = apply_policy(base, row, policy)
        baseline_rows.append(base)
        learned_rows.append(learned)
        y.append(OUTCOME_INDEX[str(row["actual_outcome"])])
        policy_sources.append(source)

    yy = np.asarray(y, dtype=int)
    baseline = np.vstack(baseline_rows)
    learned = np.vstack(learned_rows)

    baseline_metrics = _metrics(yy, baseline)
    learned_metrics = _metrics(yy, learned)
    improvement = {
        "Accuracy": learned_metrics["accuracy"] - baseline_metrics["accuracy"],
        "LogLoss": baseline_metrics["logloss"] - learned_metrics["logloss"],
        "Brier": baseline_metrics["brier"] - learned_metrics["brier"],
    }

    uncertainty: dict[str, Any]
    if len(yy) >= bootstrap_min_cases:
        uncertainty = uncertainty_to_dict(
            paired_block_bootstrap(
                y=yy,
                baseline_proba=baseline,
                candidate_proba=learned,
                block_size=min(30, max(5, len(yy) // 4)),
                replications=400,
                seed=42,
            )
        )
    else:
        uncertainty = {
            "status": "UNAVAILABLE",
            "reason": "holdout too small for paired block bootstrap",
            "rows": int(len(yy)),
            "required_rows": int(bootstrap_min_cases),
        }

    # Deliberately conservative candidate gate. A research candidate is not
    # promoted just for a point estimate; each primary direction must improve,
    # while uncertainty remains visible. With unavailable uncertainty we HOLD.
    primary_nonworse = (
        improvement["LogLoss"] >= 0.0
        and improvement["Brier"] >= 0.0
        and improvement["Accuracy"] >= 0.0
    )
    statistically_supported = (
        uncertainty.get("status") == "EVALUATED"
        and uncertainty.get("p_improvement_positive", {}).get("LogLoss", 0.0) >= 0.80
        and uncertainty.get("p_improvement_positive", {}).get("Brier", 0.0) >= 0.80
    )
    promotion_status = (
        "PROMOTION_CANDIDATE"
        if primary_nonworse and statistically_supported
        else "HOLD"
    )

    return {
        "status": "EVALUATED",
        "promotion_status": promotion_status,
        "rows": int(n),
        "train_cases": int(len(train)),
        "holdout_cases": int(len(holdout)),
        "holdout_cutoff_utc": holdout_cutoff.isoformat(),
        "policy_matured_rows": int(policy.get("matured_rows", 0)),
        "policy_hash": policy.get("policy_hash"),
        "policy_coverage": float(np.mean([x is not None for x in policy_sources])),
        "baseline": baseline_metrics,
        "learned": learned_metrics,
        "improvement": improvement,
        "uncertainty": uncertainty,
        "selection_contract": {
            "holdout_outcomes_used_to_build_policy": False,
            "holdout_used_for_parameter_selection": False,
            "fixed_policy_hyperparameters": True,
            "production_modified": False,
            "auto_promotion": False,
        },
    }


def run(input_path: str | Path = DEFAULT_INPUT) -> dict[str, Any]:
    frame = load_experience(input_path)
    result = evaluate_locked_holdout(frame)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "experience_learning_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    args = parser.parse_args()
    result = run(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
