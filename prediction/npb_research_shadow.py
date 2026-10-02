"""Explicit on-demand NPB research-shadow prediction entry point.

This is never a production fallback. It can request any pregame lead-time
window and records the result in the research-shadow scope.
"""
from __future__ import annotations

import argparse
import json

from production_npb import predict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, JST")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--minimum-lead-minutes", type=float, default=0.0)
    parser.add_argument("--maximum-lead-minutes", type=float, default=None)
    parser.add_argument("--preferred-lead-minutes", type=float, default=60.0)
    args = parser.parse_args()

    if args.minimum_lead_minutes < 0.0:
        raise SystemExit("--minimum-lead-minutes must be >= 0")
    if (
        args.maximum_lead_minutes is not None
        and args.maximum_lead_minutes < args.minimum_lead_minutes
    ):
        raise SystemExit("--maximum-lead-minutes must be >= minimum")

    result = predict(
        args.date,
        args.data_dir,
        research_shadow=True,
        minimum_lead_minutes=args.minimum_lead_minutes,
        maximum_lead_minutes=args.maximum_lead_minutes,
        preferred_lead_minutes=args.preferred_lead_minutes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("execution_status") in {
        "RESEARCH_SHADOW_EXECUTED",
        "NO_FUTURE_GAMES",
        "BLOCKED_STARTERS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
