"""Strict verifier for score-distribution research artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.score_distribution_pattern_lab import (
    MEAN_SHRINKS,
    MODEL_MIXES,
    SHARED_SCALES,
)


def validate(path: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema_version") != "baseball-score-distribution-pattern-lab-v1":
        raise RuntimeError("score lab schema mismatch")
    if obj.get("status") != "RESEARCH_ONLY":
        raise RuntimeError("artifact is not research-only")
    if obj.get("decision") != "NO_AUTO_ADOPTION":
        raise RuntimeError("artifact permits auto-adoption")
    sha = str(obj.get("git_commit_sha", ""))
    if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
        raise RuntimeError("artifact Git snapshot SHA is missing or invalid")

    catalog = obj.get("variant_catalog", {})
    expected = len(MODEL_MIXES) * len(SHARED_SCALES) * len(MEAN_SHRINKS)
    if int(catalog.get("variant_count", 0)) != expected:
        raise RuntimeError("score variant count mismatch")
    if catalog.get("model_mixes") != [x[0] for x in MODEL_MIXES]:
        raise RuntimeError("score model-mix catalog mismatch")
    if catalog.get("shared_scales") != list(SHARED_SCALES):
        raise RuntimeError("score shared-scale catalog mismatch")
    if catalog.get("mean_shrinks") != list(MEAN_SHRINKS):
        raise RuntimeError("score mean-shrink catalog mismatch")

    winner = obj.get("winner", {})
    if not str(winner.get("variant_id", "")).strip():
        raise RuntimeError("score winner missing")
    rows = obj.get("development_top", [])
    if not rows:
        raise RuntimeError("development score-pattern evidence is empty")
    stability = obj.get("rank_stability", {})
    if stability.get("eligible") is not True:
        raise RuntimeError("score rank-stability evidence is missing")
    if not str(stability.get("best_variant", "")).strip():
        raise RuntimeError("score rank-stability best variant is missing")

    holdout = obj.get("locked_holdout", {})
    if holdout.get("winner_only") is not True:
        raise RuntimeError("score holdout must be winner-only")
    if holdout.get("variant_id") != winner.get("variant_id"):
        raise RuntimeError("score holdout variant differs from selected winner")
    if int(obj.get("locked_holdout_rows", 0)) < 1:
        raise RuntimeError("score holdout is empty")
    metrics = holdout.get("metrics", {})
    for key in (
        "ScoreMAE",
        "HighLogLoss",
        "HighBrier",
        "HighAccuracy",
        "Top4HitRate",
        "ExactScoreRows",
        "rows",
    ):
        if key not in metrics:
            raise RuntimeError(f"score holdout metric missing: {key}")
    if int(metrics["rows"]) != int(obj["locked_holdout_rows"]):
        raise RuntimeError("score holdout row count does not reconcile")

    gate = obj.get("research_gate", {})
    if gate.get("holdout_locked_before_selection") is not True:
        raise RuntimeError("score holdout lock evidence missing")
    if gate.get("no_auto_adoption") is not True:
        raise RuntimeError("score no-auto-adoption evidence missing")
    if int(gate.get("unexpected_execution_failures", 0)) != 0:
        raise RuntimeError("unexpected score execution failures exist")

    folds = obj.get("folds", {})
    names = ("screen", "confirm", "deep")
    for idx, name in enumerate(names):
        fold = folds.get(name, {})
        start = int(fold.get("start_row", -1))
        end = int(fold.get("end_row", -1))
        if end <= start:
            raise RuntimeError(f"invalid chronological score fold: {name}")
        if idx and start < int(folds[names[idx - 1]].get("end_row", -1)):
            raise RuntimeError("score chronological folds overlap or regress")

    return {
        "league": obj.get("league"),
        "status": obj.get("status"),
        "variant_count": catalog.get("variant_count"),
        "winner": winner.get("variant_id"),
        "rank_stability": stability,
        "holdout_rows": obj.get("locked_holdout_rows"),
        "holdout": metrics,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    a = p.parse_args()
    print(json.dumps(validate(a.path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
