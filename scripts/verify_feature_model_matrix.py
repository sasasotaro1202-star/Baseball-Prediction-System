"""Validate a feature-model matrix artifact without changing its decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "status": "RESEARCH_ONLY",
        "decision": "NO_AUTO_ADOPTION",
        "selection_basis": "Development OOS only",
    }
    for key, expected in required.items():
        if obj.get(key) != expected:
            raise RuntimeError(f"{key}={obj.get(key)!r}; expected {expected!r}")
    if obj.get("holdout_locked_before_selection") is not True:
        raise RuntimeError("locked holdout contract is missing or false")
    from research.feature_set_variants import SCREENING_VARIANTS

    requested = int(obj.get("matrix_size_requested", 0))
    executed = int(obj.get("matrix_size_executed", 0))
    variants = list(obj.get("variants_requested", []))
    half_lives = list(obj.get("half_lives_requested", []))
    pools = list(obj.get("model_pools_requested", []))
    if variants != list(SCREENING_VARIANTS):
        raise RuntimeError("requested feature variants do not match canonical registry")
    if len(variants) != len(set(variants)):
        raise RuntimeError("duplicate feature variants in matrix request")
    if len(half_lives) != 5:
        raise RuntimeError("matrix must evaluate exactly five recency half-lives")
    if requested != len(variants) * len(half_lives) * max(1, len(pools)):
        raise RuntimeError("matrix requested-size arithmetic is inconsistent")
    if executed != requested:
        raise RuntimeError(f"matrix execution incomplete: executed={executed}, requested={requested}")
    successful = int(obj.get("successful_configs", 0))
    failed = int(obj.get("failed_configs", 0))
    blocked = int(obj.get("blocked_pit_context_configs", 0))
    execution_failed = int(obj.get("execution_failed_configs", 0))
    if successful < 1:
        raise RuntimeError("no successful configurations")
    if successful + failed != executed:
        raise RuntimeError("matrix success/failure counts do not reconcile with execution count")
    if failed != blocked + execution_failed:
        raise RuntimeError("matrix failure taxonomy does not reconcile")
    if execution_failed != 0:
        raise RuntimeError("one or more matrix configurations failed execution")
    if int(obj.get("locked_holdout_rows", 0)) < 1:
        raise RuntimeError("locked holdout is empty")
    winner = obj.get("winner")
    holdout = obj.get("locked_holdout", {})
    if holdout.get("winner_only") is not True:
        raise RuntimeError("locked holdout must be winner-only")
    holdout_metrics = holdout.get("metrics", {})
    for key in ("LogLoss", "Brier", "Accuracy", "ECE", "rows"):
        if key not in holdout_metrics:
            raise RuntimeError(f"locked holdout metric missing: {key}")
    if not isinstance(winner, dict) or not str(winner.get("config_id", "")).strip():
        raise RuntimeError("Development-selected winner is missing")
    return {
        "league": obj.get("league"),
        "status": obj.get("status"),
        "matrix_requested": obj.get("matrix_size_requested"),
        "matrix_executed": obj.get("matrix_size_executed"),
        "successful": obj.get("successful_configs"),
        "failed": obj.get("failed_configs"),
        "winner": winner.get("config_id"),
        "holdout_rows": obj.get("locked_holdout_rows"),
        "blocked_pit_context": obj.get("blocked_pit_context_configs", 0),
        "execution_failed": obj.get("execution_failed_configs", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(validate(args.path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
