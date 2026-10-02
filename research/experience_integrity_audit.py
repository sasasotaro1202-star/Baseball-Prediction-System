"""Independent integrity audit for immutable production prediction snapshots.

The audit is read-only and fail-closed. It checks provenance/timing, PIT status,
probability normalization, and prediction-id uniqueness before experience data
can be treated as trustworthy research input.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.experience_ledger import _is_legacy_scheduled_cutoff

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED_DIR = ROOT / "data" / "experience" / "predictions"
REQUIRED_FIELDS = (
    "game_id",
    "datetime_jst",
    "prediction_cutoff_utc",
    "prediction_generated_at",
    "starter_evidence_observed_at_utc",
    "starter_evidence_status",
    "pit_status",
)
STARTER_EVIDENCE_STATUSES = {"official_announced", "official_announced_snapshot"}


def _parse_ts(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {field}")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {field}") from exc
    if dt.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return dt.astimezone(timezone.utc)


def _check_probabilities(row: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    values: list[float] = []
    for field in fields:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid {label} probability field: {field}") from exc
        if not math.isfinite(value) or value < 0.0 or value > 100.0:
            raise ValueError(f"invalid {label} probability value: {value!r}")
        values.append(value)
    total = sum(values)
    if abs(total - 100.0) > 0.0003:
        raise ValueError(f"invalid {label} probability sum: {total:.8f}")


def audit_prediction_directory(pred_dir: str | Path = DEFAULT_PRED_DIR) -> dict[str, Any]:
    root = Path(pred_dir)
    if not root.exists():
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_head": str(__import__("os").environ.get("GITHUB_SHA") or "unknown"),
            "status": "NO_PREDICTIONS",
            "prediction_files": 0,
            "prediction_rows": 0,
            "pit_pass_rows": 0,
        }

    files = sorted(root.glob("*.jsonl"))
    seen_ids: set[str] = set()
    rows = 0
    pit_pass = 0
    legacy_quarantined = 0

    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"prediction row is not an object: {path}:{line_no}")

            missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
            if missing:
                raise ValueError(
                    f"prediction row missing PIT/provenance fields: {path}:{line_no}: {missing}"
                )

            is_legacy = _is_legacy_scheduled_cutoff(row)
            game_time = _parse_ts(row["datetime_jst"], field="datetime_jst")
            cutoff = _parse_ts(row["prediction_cutoff_utc"], field="prediction_cutoff_utc")
            generated = _parse_ts(row["prediction_generated_at"], field="prediction_generated_at")
            observed = _parse_ts(
                row["starter_evidence_observed_at_utc"],
                field="starter_evidence_observed_at_utc",
            )

            prediction_id = str(row.get("prediction_id", "")).strip()
            if not prediction_id:
                raise ValueError(f"prediction row missing prediction_id: {path}:{line_no}")
            if prediction_id in seen_ids:
                raise ValueError(f"duplicate prediction_id: {prediction_id}")
            seen_ids.add(prediction_id)

            if is_legacy:
                legacy_quarantined += 1
                continue

            if not cutoff < game_time:
                raise ValueError(f"prediction cutoff is not pregame: {path}:{line_no}")
            if not cutoff <= generated < game_time:
                raise ValueError(f"prediction generation timing is invalid: {path}:{line_no}")
            if observed > cutoff:
                raise ValueError(f"starter evidence observed after cutoff: {path}:{line_no}")
            if str(row["starter_evidence_status"]).strip() not in STARTER_EVIDENCE_STATUSES:
                raise ValueError(f"unsupported starter evidence status: {path}:{line_no}")
            starter_source = str(row.get("starter_source") or "").strip()
            if starter_source and not starter_source.startswith("https://npb.jp/"):
                raise ValueError(f"non-official starter source: {path}:{line_no}")
            if str(row["pit_status"]).upper() != "PASS":
                raise ValueError(f"prediction is not PIT PASS: {path}:{line_no}")
            pit_pass += 1

            _check_probabilities(
                row,
                ("home_win_pct", "draw_pct", "away_win_pct"),
                "win",
            )
            _check_probabilities(row, ("low_pct", "high_pct"), "Low/High")

            rows += 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": str(__import__("os").environ.get("GITHUB_SHA") or "unknown"),
        "status": "PASS_WITH_LEGACY_QUARANTINE" if legacy_quarantined else "PASS",
        "prediction_files": len(files),
        "prediction_rows": rows,
        "pit_pass_rows": pit_pass,
        "legacy_quarantined_rows": legacy_quarantined,
        "unique_prediction_ids": len(seen_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, default=str(DEFAULT_PRED_DIR))
    args = parser.parse_args()
    result = audit_prediction_directory(args.pred_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
