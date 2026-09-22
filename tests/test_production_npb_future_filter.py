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
            },
            {
                "home": "C", "away": "D",
                "home_starter": "H2", "away_starter": "A2",
                "confirmed_starters": True,
                "starter_evidence_status": "official_announced",
                "official_start_time": "18:00",
            },
        ],
    )
    monkeypatch.setattr(
        production_npb,
        "_utc_now",
        lambda: pd.Timestamp("2026-09-22T00:00:00Z"),
    )

    out = production_npb.build_target_rows("2026-09-22")
    assert out["home"].tolist() == ["C"]
    assert out["away"].tolist() == ["D"]
