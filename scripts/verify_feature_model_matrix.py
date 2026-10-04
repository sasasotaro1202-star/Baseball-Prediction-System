"""Verify a feature/model matrix artifact under the research-only contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from research.feature_set_variants import SCREENING_VARIANTS


def verify(path: str, pool: str) -> dict[str, object]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    assert obj.get("status") == "RESEARCH_ONLY"
    assert obj.get("decision") == "NO_AUTO_ADOPTION"
    assert obj.get("holdout_locked_before_selection") is True
    assert int(obj.get("matrix_size_executed", 0)) > 0
    assert len(obj.get("variants_requested", [])) == len(SCREENING_VARIANTS)
    assert len(obj.get("half_lives_requested", [])) == 5
    assert obj.get("model_pools_requested") == [pool]
    assert int(obj.get("matrix_size_requested", 0)) == len(SCREENING_VARIANTS) * 5
    failed = int(obj.get("failed_configs", 0))
    blocked = int(obj.get("blocked_pit_context_configs", 0))
    exec_failed = int(obj.get("execution_failed_configs", 0))
    assert failed == blocked + exec_failed
    assert exec_failed == 0
    return {
        "league": obj.get("league"),
        "pool": pool,
        "variants": len(obj.get("variants_requested", [])),
        "matrix_requested": obj.get("matrix_size_requested"),
        "matrix_executed": obj.get("matrix_size_executed"),
        "successful": obj.get("successful_configs"),
        "blocked_pit_context": blocked,
        "winner": obj.get("winner", {}).get("config_id"),
        "holdout_rows": obj.get("locked_holdout_rows"),
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_feature_model_matrix.py <json_path> <model_pool>")
    print(json.dumps(verify(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
