from __future__ import annotations

from datetime import datetime, timezone
import json

import scripts.npb_rapid_pregame as rapid


def test_rapid_slot_selects_only_60_to_90_minutes(monkeypatch):
    monkeypatch.setattr(
        rapid.scheduler,
        "_schedule_for_date",
        lambda target_date: [
            {
                "home": "東北楽天ゴールデンイーグルス",
                "away": "福岡ソフトバンクホークス",
                "official_start_time": "18:00",
            },
            {
                "home": "千葉ロッテマリーンズ",
                "away": "埼玉西武ライオンズ",
                "official_start_time": "19:30",
            },
        ],
    )
    now = datetime(2026, 10, 5, 7, 30, tzinfo=timezone.utc)  # 16:30 JST
    out = rapid.rapid_due_games(now_utc=now)
    assert out["status"] == "DUE"
    assert len(out["due_games"]) == 1
    row = out["due_games"][0]
    assert row["game_id"] == "NPB-2026-10-05-1"
    assert row["home"] == "東北楽天ゴールデンイーグルス"
    assert row["prediction_source"] == "RESEARCH_SHADOW_AUTO_90M"
    assert 60.0 < row["lead_minutes"] <= 90.0


def test_rapid_slot_deduplicates_existing_source(monkeypatch, tmp_path):
    pred_dir = tmp_path / "data" / "experience" / "research_shadow" / "predictions"
    pred_dir.mkdir(parents=True)
    (pred_dir / "2026-10-05.jsonl").write_text(
        json.dumps(
            {
                "game_id": "NPB-2026-10-05-1",
                "prediction_source": "RESEARCH_SHADOW_AUTO_90M",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rapid, "ROOT", tmp_path)
    monkeypatch.setattr(
        rapid.scheduler,
        "_schedule_for_date",
        lambda target_date: [
            {
                "home": "東北楽天ゴールデンイーグルス",
                "away": "福岡ソフトバンクホークス",
                "official_start_time": "18:00",
            }
        ],
    )
    now = datetime(2026, 10, 5, 7, 20, tzinfo=timezone.utc)
    out = rapid.rapid_due_games(now_utc=now)
    assert out["due_games"] == []
    assert out["status"] == "NO_DUE_GAMES"


def test_rapid_slot_fail_closes_invalid_window():
    try:
        rapid.rapid_due_games(
            now_utc=datetime(2026, 10, 5, 7, 20, tzinfo=timezone.utc),
            min_lead_minutes=90,
            max_lead_minutes=60,
        )
    except ValueError:
        return
    raise AssertionError("invalid rapid pregame lead window must fail closed")
