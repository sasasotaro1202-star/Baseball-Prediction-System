"""Execute the real-data Baseball candidate lifecycle without self-comparison.

The existing BaseballBacktest remains the numerical source of truth. Candidate
selection occurs only on chronological Development OOS; the later holdout is
never used to select a candidate. Missing PIT/market evidence remains a hard
block rather than being replaced with synthetic evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research.npb_candidate_replay import run_npb_candidate_cycle
from research.mlb_candidate_replay import run_mlb_candidate_cycle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class RealDataRun:
    league: str
    decision: str
    stage: str
    source_rows: int | None
    development_rows: int | None
    holdout_rows: int | None
    reasons: tuple[str, ...]
    evidence: dict[str, Any]


def _run_npb(data_dir: str, commit: str) -> RealDataRun:
    try:
        result = run_npb_candidate_cycle(data_dir=data_dir, git_commit=commit)
    except Exception as exc:
        return RealDataRun("NPB", "HOLD", "error", None, None, None,
                           (f"{type(exc).__name__}: {exc}",), {})
    holdout = result.get("holdout", {}) if isinstance(result, dict) else {}
    development = result.get("development", {}) if isinstance(result, dict) else {}
    dev_rows = max((int(v.get("rows", 0)) for v in development.values()
                    if isinstance(v, dict)), default=0)
    return RealDataRun(
        "NPB", str(result.get("decision", "HOLD")),
        str(result.get("stage", "unknown")), None, dev_rows,
        int(holdout.get("holdout_rows", 0)) if holdout else None,
        tuple(result.get("reasons", ())), result,
    )


def _run_mlb(data_dir: str, commit: str, start: int, end: int) -> RealDataRun:
    try:
        result = run_mlb_candidate_cycle(
            data_dir=data_dir, git_commit=commit,
            mlb_start=start, mlb_end=end,
        )
    except Exception as exc:
        return RealDataRun("MLB", "HOLD", "error", None, None, None,
                           (f"{type(exc).__name__}: {exc}",), {})
    holdout = result.get("holdout", {}) if isinstance(result, dict) else {}
    baseline = holdout.get("baseline", {}) if isinstance(holdout, dict) else {}
    return RealDataRun(
        "MLB", str(result.get("decision", "HOLD")),
        str(result.get("stage", "unknown")), None, None,
        int(baseline.get("rows", 0)) if baseline else None,
        tuple(result.get("reasons", ())), result,
    )


def run_real_data_validation(*, data_dir: str | Path = "data",
                             league: str | None = None,
                             mlb_start: int = 2020,
                             mlb_end: int = 2026,
                             git_commit: str | None = None) -> dict[str, Any]:
    """Run actual candidate replay for requested leagues; never self-compare."""
    commit = git_commit or os.getenv("GITHUB_SHA", "unknown")
    leagues = [league] if league else ["NPB", "MLB"]
    runs: list[RealDataRun] = []
    for item in leagues:
        if item == "NPB":
            runs.append(_run_npb(str(data_dir), commit))
        elif item == "MLB":
            runs.append(_run_mlb(str(data_dir), commit, mlb_start, mlb_end))
        else:
            raise ValueError("league must be NPB, MLB, or omitted")

    decisions = [r.decision for r in runs]
    overall = "ADOPT" if runs and all(d == "ADOPT" for d in decisions) else "HOLD"
    payload = {
        "status": "COMPLETED_WITH_FAIL_CLOSED_DECISION",
        "overall_decision": overall,
        "runs": [asdict(r) for r in runs],
        "rules": {
            "candidate_distinct_from_baseline": True,
            "development_oos_before_candidate_lock": True,
            "independent_locked_holdout": True,
            "missing_score_or_hilo_or_pit_line_evidence_blocks_adoption": True,
            "hardcoded_total_run_threshold_is_not_production_evidence": True,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "real_data_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["NPB", "MLB"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mlb-start", type=int, default=2020)
    parser.add_argument("--mlb-end", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run_real_data_validation(
        data_dir=args.data_dir, league=args.league,
        mlb_start=args.mlb_start, mlb_end=args.mlb_end,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
