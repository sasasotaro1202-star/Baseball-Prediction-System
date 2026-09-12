"""Accuracy-target governance for the Baseball research system.

The target is ambitious (80%+) but never used to manufacture a result.  This
module only evaluates measured out-of-sample metrics against the target and
keeps the target separate from model-selection mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Mapping


@dataclass(frozen=True)
class AccuracyTarget:
    name: str
    target: float = 0.80
    minimum_rows: int = 200


TARGETS = {
    "NPB_WIN": AccuracyTarget("NPB_WIN"),
    "MLB_WIN": AccuracyTarget("MLB_WIN"),
    "NPB_TOP_DRAW": AccuracyTarget("NPB_TOP_DRAW"),
    "LOW_HIGH": AccuracyTarget("LOW_HIGH"),
    "SCORE_TOP4": AccuracyTarget("SCORE_TOP4"),
}


def evaluate_target(
    target_name: str,
    *,
    accuracy: float,
    rows: int,
    independent_holdout: bool,
) -> dict[str, object]:
    """Return an auditable target assessment; never substitute missing data."""
    if target_name not in TARGETS:
        raise ValueError(f"unknown accuracy target: {target_name}")
    if not math.isfinite(float(accuracy)):
        raise ValueError("accuracy must be finite")
    if rows < 0:
        raise ValueError("rows must be non-negative")
    target = TARGETS[target_name]
    return {
        "target": asdict(target),
        "measured_accuracy": float(accuracy),
        "rows": int(rows),
        "independent_holdout": bool(independent_holdout),
        "target_reached": bool(rows >= target.minimum_rows and independent_holdout and accuracy >= target.target),
        "status": "TARGET_REACHED" if rows >= target.minimum_rows and independent_holdout and accuracy >= target.target else "BELOW_TARGET_OR_UNVERIFIED",
    }


def evaluate_targets(metrics: Mapping[str, Mapping[str, object]], *, independent_holdout: bool) -> dict[str, dict[str, object]]:
    """Evaluate all supplied targets without inventing absent measurements."""
    result: dict[str, dict[str, object]] = {}
    for name, values in metrics.items():
        if name not in TARGETS:
            continue
        if "accuracy" not in values or "rows" not in values:
            result[name] = {"status": "UNVERIFIED", "reason": "accuracy_or_rows_missing"}
            continue
        result[name] = evaluate_target(name, accuracy=float(values["accuracy"]), rows=int(values["rows"]), independent_holdout=independent_holdout)
    return result
