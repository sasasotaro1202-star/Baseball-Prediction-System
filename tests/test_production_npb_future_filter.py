import pandas as pd

import production_npb


def test_build_target_rows_excludes_started_games(monkeypatch):
    monkeypatch.setattr(
        production_npb,
        "official_starters",
        lambda target_date: [
            {
                "home": "A", "away": "B",
                "home_starter": "H1", "away_starter": "A1",
                "confirmed_starters": True,
                "starter_evidence_status": "official_announced",
                "official_start_time": "08:00",
                "starter_source": "https://npb.jp/",
            },
            {
                "home": "C", "away": "D",
                "home_starter": "H2", "away_starter": "A2",
                "confirmed_starters": True,
                "starter_evidence_status": "official_announced",
                "official_start_time": "18:00",
                "starter_source": "https://npb.jp/",
            },
        ],
    )
    monkeypatch.setattr(
        production_npb,
        "_utc_now",
        lambda: pd.Timestamp("2026-09-22T00:00:00Z"),
    )
    monkeypatch.setattr(
        production_npb,
        "_official_daily_start_times",
        lambda target_date: {
            ("A", "B"): "08:00",
            ("C", "D"): "18:00",
        },
    )

    out = production_npb.build_target_rows("2026-09-22")
    assert out["home"].tolist() == ["C"]
    assert out["away"].tolist() == ["D"]
