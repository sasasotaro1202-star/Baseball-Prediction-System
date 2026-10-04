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
    if int(obj.get("matrix_size_executed", 0)) < 1:
        raise RuntimeError("no matrix configurations executed")
    if int(obj.get("successful_configs", 0)) < 1:
        raise RuntimeError("no successful configurations")
    if int(obj.get("locked_holdout_rows", 0)) < 1:
        raise RuntimeError("locked holdout is empty")
    winner = obj.get("winner")
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(validate(args.path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
