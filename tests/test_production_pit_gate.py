from production_pit_gate import check_game

CUTOFF = "2026-09-19T12:00:00+00:00"


def row(level="OFFICIAL_ANNOUNCEMENT", ts="2026-09-19T10:00:00+00:00"):
    return {
        "event_id": "MLB:1",
        "league": "MLB",
        "event_start_at": "2026-09-19T15:00:00+00:00",
        "home_starter": "Home Pitcher",
        "away_starter": "Away Pitcher",
        "home_starter_evidence_level": level,
        "away_starter_evidence_level": level,
        "home_starter_announced_at": ts,
        "away_starter_announced_at": ts,
        "source": "https://npb.jp/",
    }


def test_strict_gate_accepts_two_official_announcements():
    ok, reason = check_game(row(), CUTOFF)
    assert ok and reason == "eligible"


def test_retrieval_only_is_rejected():
    ok, reason = check_game(row("RETRIEVAL_ONLY"), CUTOFF)
    assert not ok and "not_strictly_eligible" in reason


def test_non_official_source_is_rejected():
    r = row()
    r["source"] = "https://example.com/"
    ok, reason = check_game(r, CUTOFF)
    assert not ok and "not_strictly_eligible" in reason


def test_future_announcement_is_rejected():
    ok, reason = check_game(row(ts="2026-09-19T13:00:00+00:00"), CUTOFF)
    assert not ok


def test_missing_starter_is_rejected():
    r = row()
    r["away_starter"] = ""
    ok, reason = check_game(r, CUTOFF)
    assert not ok and reason == "away_starter_missing"


def test_cli_rejects_empty_evidence(tmp_path):
    import json
    import subprocess
    import sys
    src = tmp_path / "empty.json"
    src.write_text(json.dumps({"games": []}), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "production_pit_gate.py", str(src), "--cutoff", CUTOFF],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "zero games" in (proc.stderr + proc.stdout).lower()


def test_started_game_is_rejected_even_with_official_starters():
    r = row()
    r["event_start_at"] = "2026-09-19T11:00:00+00:00"
    ok, reason = check_game(r, CUTOFF)
    assert not ok and reason == "game_already_started_or_not_future_at_cutoff"


def test_missing_event_start_is_rejected():
    r = row()
    r.pop("event_start_at")
    ok, reason = check_game(r, CUTOFF)
    assert not ok and reason == "event_start_invalid"

