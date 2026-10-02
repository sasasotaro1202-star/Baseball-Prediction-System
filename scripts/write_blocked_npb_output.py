from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

# Allow direct execution as `python scripts/...py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.target_strategy import as_dict, standard_target_strategies


def main() -> None:
    target = os.environ["TARGET_DATE"]
    result = {
        "schema_version": "npb-production-v1",
        "target_date": target,
        "execution_status": "BLOCKED_PRODUCTION_GATE",
        "pit_status": "NOT_RUN",
        "starter_gate": "NOT_RUN",
        "model_status": "NOT_RUN",
        "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
        "predictions": [],
        "target_strategy_contracts": {
            key: as_dict(value) for key, value in standard_target_strategies("NPB").items()
        },
        "block_reason": "NPB production runtime is not currently eligible; external acquisition/model work was intentionally skipped.",
        "prediction_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out = Path("results") / f"npb_production_{target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
