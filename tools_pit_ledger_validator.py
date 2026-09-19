"""Strict structural validator for append-only PIT ledgers.

This does not decide whether a source is authoritative. It prevents malformed
or internally contradictory evidence from silently reaching downstream replay.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIT = ROOT / "data" / "pit"

PIT_FILENAMES = (
    "event_observations.jsonl",
    "availability_observations.jsonl",
    "source_snapshots.jsonl",
    "acquisition_runs.jsonl",
)

def pit_files():
    return [PIT / name for name in PIT_FILENAMES]

LEVELS = {"NONE", "RETRIEVAL_ONLY", "THIRD_PARTY_FIRST_SEEN", "OFFICIAL_PUBLICATION", "OFFICIAL_ANNOUNCEMENT"}


def dt(v):
    if v in (None, ""):
        return None
    x = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    if x.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return x.astimezone(timezone.utc)


def load(path):
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{n}: row must be object")
        out.append((n, row))
    return out


def validate():
    counts = {}
    for path in pit_files():
        rows = load(path)
        counts[str(path.relative_to(PIT))] = len(rows)
        for n, row in rows:
            for key in ("observed_at", "retrieved_at", "prediction_cutoff"):
                if key in row and row[key] not in (None, ""):
                    dt(row[key])
            if "available_at" in row and row["available_at"] not in (None, ""):
                available = dt(row["available_at"])
                retrieved = dt(row.get("retrieved_at"))
                if retrieved and available > retrieved:
                    raise ValueError(f"{path}:{n}: available_at after retrieved_at")
            for side in ("home", "away"):
                level_key = f"{side}_starter_evidence_level"
                ann_key = f"{side}_starter_announced_at"
                starter_key = f"{side}_starter"
                if level_key not in row:
                    continue
                level = str(row[level_key]).upper()
                if level not in LEVELS:
                    raise ValueError(f"{path}:{n}: invalid {level_key}={level}")
                ann = row.get(ann_key)
                if level == "OFFICIAL_ANNOUNCEMENT":
                    if not row.get(starter_key) or not ann:
                        raise ValueError(f"{path}:{n}: official announcement requires starter and timestamp")
                    cutoff = dt(row.get("prediction_cutoff"))
                    announcement = dt(ann)
                    if cutoff and announcement > cutoff:
                        raise ValueError(f"{path}:{n}: announcement after prediction cutoff")
                elif ann:
                    # Lower evidence levels may retain a publication/first-seen
                    # timestamp, but it must never be presented as an official
                    # announcement by changing the level.
                    dt(ann)
    return counts


if __name__ == "__main__":
    print(json.dumps({"status": "PIT_LEDGER_VALID", "files": validate()}, ensure_ascii=False, indent=2))
