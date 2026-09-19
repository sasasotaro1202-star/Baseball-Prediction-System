import json
from pathlib import Path

from tools_pit_ledger_validator import validate


def test_validator_rejects_official_without_timestamp(tmp_path, monkeypatch):
    import tools_pit_ledger_validator as v
    p = tmp_path / "data/pit"
    p.mkdir(parents=True)
    for name in ["event_observations.jsonl","availability_observations.jsonl","source_snapshots.jsonl","acquisition_runs.jsonl"]:
        (p/name).write_text("", encoding="utf-8")
    (p/"availability_observations.jsonl").write_text(json.dumps({
        "event_id":"MLB:1","home_starter":"A","away_starter":"B",
        "home_starter_evidence_level":"OFFICIAL_ANNOUNCEMENT",
        "away_starter_evidence_level":"RETRIEVAL_ONLY",
        "prediction_cutoff":"2026-09-19T12:00:00+00:00"
    })+"\n", encoding="utf-8")
    monkeypatch.setattr(v, "PIT", p)
    try:
        validate()
    except ValueError as e:
        assert "official announcement requires" in str(e)
    else:
        raise AssertionError("validator accepted malformed official evidence")
