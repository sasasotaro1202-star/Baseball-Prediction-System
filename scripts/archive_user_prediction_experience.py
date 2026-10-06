#!/usr/bin/env python3
"""Archive user-requested NPB research-shadow results into Shadow Experience.

This scanner is separate from production Experience. It is idempotent, preserves
request provenance, and fails closed on malformed PIT/probability evidence.
Legacy UNKNOWN competition identity is explicitly quarantined and never reused.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from research.shadow_experience import archive_shadow_output

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "prediction_requests" / "results"


def _load(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain an object")
    return obj


def _quarantine_reason(predictions: list[dict[str, Any]]) -> str | None:
    for row in predictions:
        status = str(row.get("competition_classification_status") or "").strip().lower()
        competition = str(row.get("competition") or "").strip().lower()
        stage = str(row.get("competition_stage") or "").strip().lower()
        season_type = str(row.get("season_type") or "").strip().lower()
        game_class = str(row.get("game_class") or "").strip().lower()
        key = str(row.get("competition_key") or "").strip()
        if status != "classified":
            return "UNKNOWN_IDENTITY"
        if not key or competition in {"", "unknown", "npb_unknown"} or competition.endswith("_unknown"):
            return "UNKNOWN_IDENTITY"
        if stage in {"", "unknown"} or season_type in {"", "unknown"} or game_class in {"", "unknown"}:
            return "UNKNOWN_IDENTITY"
    return None


def archive_result_file(path: Path, *, run_id: str | None = None) -> dict[str, Any]:
    outer = _load(path)
    lane = str(outer.get("generation_lane") or "").strip()
    if lane != "VALIDATED_RESEARCH_SHADOW":
        return {
            "path": str(path),
            "status": "SKIPPED_NOT_RESEARCH_SHADOW",
            "archived": 0,
            "updated": 0,
        }

    competition_id = str(outer.get("competition_id") or "").strip().upper()
    if competition_id != "NPB":
        return {
            "path": str(path),
            "status": "SKIPPED_NON_NPB",
            "archived": 0,
            "updated": 0,
        }

    output = outer.get("prediction_output")
    if not isinstance(output, dict):
        raise ValueError(f"{path}: missing prediction_output")

    execution_status = str(output.get("execution_status") or "").strip()
    if execution_status != "RESEARCH_SHADOW_EXECUTED":
        return {
            "path": str(path),
            "status": "SKIPPED_NON_EXECUTED",
            "execution_status": execution_status,
            "archived": 0,
            "updated": 0,
        }

    predictions = output.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"{path}: RESEARCH_SHADOW_EXECUTED contains no predictions")
    if output.get("production_eligibility") is not False:
        raise ValueError(f"{path}: research shadow is not explicitly production-ineligible")
    if str(output.get("scope") or "").strip().upper() != "RESEARCH_SHADOW":
        raise ValueError(f"{path}: research shadow scope is not explicit")

    typed_predictions = [x for x in predictions if isinstance(x, dict)]
    if len(typed_predictions) != len(predictions):
        raise ValueError(f"{path}: prediction list contains a non-object row")
    quarantine = _quarantine_reason(typed_predictions)
    if quarantine:
        return {
            "path": str(path),
            "status": "QUARANTINED",
            "reason": quarantine,
            "request_id": str(outer.get("request_id") or ""),
            "archived": 0,
            "updated": 0,
        }

    payload = dict(output)
    payload["competition_id"] = competition_id
    request_id = str(outer.get("request_id") or "").strip() or None

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp.write("\n")
        temp_path = Path(tmp.name)

    try:
        result = archive_shadow_output(
            temp_path,
            run_id=run_id,
            request_id=request_id,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "path": str(path),
        "status": "ARCHIVED",
        "request_id": request_id,
        **result,
    }


def scan(result_dir: Path, *, run_id: str | None = None) -> dict[str, Any]:
    if not result_dir.exists():
        return {
            "schema_version": "npb-user-prediction-shadow-archive-v1",
            "status": "NO_RESULT_DIRECTORY",
            "fail_closed": True,
            "result_files_scanned": 0,
            "archived": 0,
            "updated": 0,
            "quarantined": 0,
            "skipped": 0,
            "source_head": os.environ.get("GITHUB_SHA") or "unknown",
        }

    files = sorted(result_dir.glob("*.json"))
    results: list[dict[str, Any]] = []
    totals = {"archived": 0, "updated": 0, "quarantined": 0, "skipped": 0}
    for path in files:
        outcome = archive_result_file(path, run_id=run_id)
        results.append(outcome)
        totals["archived"] += int(outcome.get("archived", 0))
        totals["updated"] += int(outcome.get("updated", 0))
        if outcome.get("status") == "QUARANTINED":
            totals["quarantined"] += 1
        elif str(outcome.get("status", "")).startswith("SKIPPED"):
            totals["skipped"] += 1

    return {
        "schema_version": "npb-user-prediction-shadow-archive-v1",
        "status": "UPDATED",
        "fail_closed": True,
        "result_files_scanned": len(files),
        **totals,
        "results": results,
        "source_head": os.environ.get("GITHUB_SHA") or "unknown",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    args = parser.parse_args()
    print(json.dumps(scan(Path(args.result_dir), run_id=args.run_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
