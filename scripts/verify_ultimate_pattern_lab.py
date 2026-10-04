"""Strict verifier for the research-only ultimate pattern lab artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.ultimate_pattern_lab import (
    CORE_HORIZONS,
    HALF_LIVES,
    OPTIONAL_FAMILIES,
    STAGE_POOLS,
    _family_pattern_catalog,
)


def validate(path: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    sha = str(obj.get("git_commit_sha", ""))
    if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
        raise RuntimeError("artifact Git snapshot SHA is missing or invalid")
    if obj.get("status") != "RESEARCH_ONLY":
        raise RuntimeError("artifact is not research-only")
    if obj.get("decision") != "NO_AUTO_ADOPTION":
        raise RuntimeError("artifact permits auto-adoption")
    counts = obj.get("stage_counts", {})
    if counts.get("stage_a_requested") != 256:
        raise RuntimeError("Stage A must request exactly 256 family patterns")
    expected_b = 8 * len(CORE_HORIZONS) * len(HALF_LIVES) * len(STAGE_POOLS)
    if counts.get("stage_b_requested") != expected_b:
        raise RuntimeError("Stage B requested-count mismatch")
    if counts.get("stage_c_requested") != 4:
        raise RuntimeError("Stage C requested-count mismatch")
    if int(counts.get("stage_a_successful", 0)) + int(counts.get("stage_a_failed", 0)) != 256:
        raise RuntimeError("Stage A success/failure does not reconcile")
    if int(counts.get("stage_b_successful", 0)) + int(counts.get("stage_b_failed", 0)) != expected_b:
        raise RuntimeError("Stage B success/failure does not reconcile")
    if int(counts.get("stage_c_successful", 0)) + int(counts.get("stage_c_failed", 0)) != 4:
        raise RuntimeError("Stage C success/failure does not reconcile")
    gate = obj.get("research_gate", {})
    if gate.get("holdout_locked_before_selection") is not True:
        raise RuntimeError("holdout lock evidence missing")
    if gate.get("no_auto_adoption") is not True:
        raise RuntimeError("no-auto-adoption gate missing")
    if int(gate.get("unexpected_execution_failures", 0)) != 0:
        raise RuntimeError("unexpected execution failures exist")
    holdout = obj.get("locked_holdout", {})
    if holdout.get("winner_only") is not True:
        raise RuntimeError("holdout must be winner-only")
    if int(obj.get("locked_holdout_rows", 0)) < 1:
        raise RuntimeError("locked holdout is empty")
    metrics = holdout.get("metrics", {})
    for key in ("LogLoss", "Brier", "Accuracy", "ECE", "rows"):
        if key not in metrics:
            raise RuntimeError(f"holdout metric missing: {key}")
    catalog = obj.get("feature_family_catalog", {})
    if catalog.get("optional_families") != list(OPTIONAL_FAMILIES):
        raise RuntimeError("feature family catalog does not match canonical lab")
    if int(catalog.get("catalog_size", 0)) != len(_family_pattern_catalog()):
        raise RuntimeError("feature family catalog size mismatch")
    winner = obj.get("winner")
    if not isinstance(winner, dict) or not str(winner.get("candidate_id", "")).strip():
        raise RuntimeError("winner missing")
    stage_c = obj.get("stage_c_all", [])
    stage_c_ids = {str(row.get("candidate_id", "")) for row in stage_c if isinstance(row, dict)}
    if str(winner["candidate_id"]) not in stage_c_ids:
        raise RuntimeError("winner is not present in Stage C evidence")
    winner_meta = winner.get("feature_meta", {})
    holdout_meta = obj.get("locked_holdout", {}).get("feature_meta", {})
    if winner_meta.get("feature_schema_hash") and holdout_meta.get("feature_schema_hash") != winner_meta.get("feature_schema_hash"):
        raise RuntimeError("holdout feature schema does not match the selected winner")
    if int(obj.get("locked_holdout", {}).get("metrics", {}).get("rows", 0)) != int(obj.get("locked_holdout_rows", 0)):
        raise RuntimeError("holdout row count does not reconcile with locked_holdout_rows")
    for name in ("screen", "confirm", "deep"):
        fold = obj.get("folds", {}).get(name, {})
        if int(fold.get("end_row", 0)) <= int(fold.get("start_row", 0)):
            raise RuntimeError(f"invalid {name} chronological fold")
    return {
        "league": obj.get("league"),
        "status": obj.get("status"),
        "stage_counts": counts,
        "winner": winner.get("candidate_id"),
        "holdout_rows": obj.get("locked_holdout_rows"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(validate(args.path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
