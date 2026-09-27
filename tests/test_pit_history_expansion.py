import json
from pathlib import Path

from core.pit_snapshot import append_snapshot, make_snapshot
from research.pit_history_expansion import audit


def test_pit_history_audit_counts_expanded_competitions(tmp_path, monkeypatch):
    pit = tmp_path / "data/pit/source_snapshots.jsonl"
    append_snapshot(make_snapshot(
        event_id="KBO:1", league="KBO", entity_type="game", entity_id="1",
        source="kbo", payload={"x": 1},
        prediction_cutoff="2026-09-28T10:00:00+00:00",
        available_at="2026-09-28T09:00:00+00:00",
        retrieved_at="2026-09-28T09:05:00+00:00",
    ), pit)
    append_snapshot(make_snapshot(
        event_id="CPBL:1", league="CPBL", entity_type="game", entity_id="1",
        source="cpbl", payload={"x": 1},
        prediction_cutoff="2026-09-28T10:00:00+00:00",
        available_at="2026-09-28T11:00:00+00:00",
        retrieved_at="2026-09-28T11:05:00+00:00",
    ), pit)
    monkeypatch.setattr("research.pit_history_expansion.PIT_FILE", pit)
    out = audit()
    assert out["replayable_rows"] == 1
    assert out["replayable_by_league"] == {"KBO": 1}
    json.dumps(out, ensure_ascii=False)
