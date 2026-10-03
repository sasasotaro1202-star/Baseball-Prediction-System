from __future__ import annotations

from datetime import date

from scripts import verify_npb_live_sources as health


def test_select_pregame_probe_skips_empty_day_without_masking_source_failure(monkeypatch):
    calls = []

    def fake_collect(target_date: str):
        calls.append(target_date)
        if target_date == "2026-10-04":
            return {"status": "AVAILABLE", "games": [], "sources": []}
        return {"status": "AVAILABLE", "games": [{"game_id": "G-1"}], "sources": []}

    monkeypatch.setattr(health, "collect_npb_pregame_context", fake_collect)
    probe_date, context = health._select_pregame_probe(date(2026, 10, 4), max_future_days=2)
    assert calls == ["2026-10-04", "2026-10-05"]
    assert probe_date == "2026-10-05"
    assert context["games"][0]["game_id"] == "G-1"


def test_select_pregame_probe_fails_closed_on_source_failure(monkeypatch):
    def fake_collect(target_date: str):
        return {
            "status": "SOURCE_FAILED",
            "games": [],
            "sources": [],
            "error": "synthetic source failure",
        }

    monkeypatch.setattr(health, "collect_npb_pregame_context", fake_collect)
    try:
        health._select_pregame_probe(date(2026, 10, 4), max_future_days=2)
    except RuntimeError as exc:
        assert "synthetic source failure" in str(exc)
    else:
        raise AssertionError("SOURCE_FAILED was not propagated fail-closed")
