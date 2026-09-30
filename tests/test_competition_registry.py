from __future__ import annotations

import pytest

from data.competition_registry import COMPETITIONS, get, production_eligible


def test_registry_contains_required_competitions() -> None:
    ids = {spec.competition_id for spec in COMPETITIONS}
    assert {
        "NPB",
        "MLB",
        "WBC",
        "WBSC_PREMIER12",
        "OLYMPICS_BASEBALL",
        "ASIAN_GAMES_BASEBALL",
        "WBSC_U18",
        "WBSC_U23",
        "KOSHIEN_SENBATSU",
        "KOSHIEN_SUMMER",
        "KOSHIEN_QUALIFIERS",
        "JAPAN_UNIVERSITY_BASEBALL",
    } <= ids


def test_registry_never_promotes_by_listing_alone() -> None:
    # NPB is fail-closed until explicit candidate/holdout adoption evidence.
    assert production_eligible("NPB") is False
    for competition_id in (
        "MLB",
        "WBC",
        "WBSC_PREMIER12",
        "OLYMPICS_BASEBALL",
        "ASIAN_GAMES_BASEBALL",
        "WBSC_U18",
        "WBSC_U23",
        "KOSHIEN_SENBATSU",
        "KOSHIEN_SUMMER",
        "KOSHIEN_QUALIFIERS",
        "JAPAN_UNIVERSITY_BASEBALL",
    ):
        assert production_eligible(competition_id) is False


def test_league_and_non_league_phases_are_explicit() -> None:
    npb = get("NPB")
    assert npb.phase_type == "league"
    assert npb.rules_profile == "npb"
    assert get("ASIAN_GAMES_BASEBALL").phase_type == "tournament"
    assert get("WBC").phase_type == "tournament"
    assert get("KOSHIEN_QUALIFIERS").phase_type == "qualifier"
    assert get("JAPAN_UNIVERSITY_BASEBALL").phase_type == "league_and_tournament"


def test_unknown_competition_fails_closed() -> None:
    with pytest.raises(KeyError):
        get("UNKNOWN_COMPETITION")



def test_competition_strategy_profiles_are_phase_specific():
    from research.competition_strategy import strategy_for, eligible_for_competition_calibration

    regular = strategy_for("NPB:npb_regular:regular_season")
    interleague = strategy_for("NPB:npb_interleague:interleague")
    postseason = strategy_for("NPB:npb_climax_series:climax_series")
    tournament = strategy_for("WBC:wbc:tournament")
    unknown = strategy_for("NPB:unknown:unknown")

    assert regular.strategy_id == "league_adaptive_ensemble"
    assert interleague.strategy_id == "league_adaptive_ensemble"
    assert postseason.strategy_id == "postseason_shrunk_ensemble"
    assert tournament.strategy_id == "tournament_shrunk_ensemble"
    assert unknown.strategy_id == "unknown_fail_closed"
    assert eligible_for_competition_calibration(regular, regular.specialist_min_validation_rows)
    assert not eligible_for_competition_calibration(unknown, 999999)



def test_competition_scope_ids_are_unique():
    from research.competition_catalog import SCOPES
    ids = [spec.scope_id for spec in SCOPES]
    assert len(ids) == len(set(ids))



def test_competition_metrics_include_top1_top4_and_lowhigh_probability_scores(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import baseball_backtest as bb

    rows = []
    for i, actual in enumerate(["3-2", "1-1"]):
        rows.append({
            "league": "NPB",
            "game_id": f"g{i}",
            "competition_key": "NPB:npb_interleague:interleague",
            "competition": "npb_interleague",
            "competition_stage": "interleague",
            "season_type": "regular_season",
            "game_class": "regular",
            "competition_classification_status": "classified",
            "prediction_strategy_id": "league_adaptive_ensemble",
            "prediction_calibration_id": "league_temperature",
            "actual": 0 if i == 0 else 1,
            "correct": 1,
            "pred_home": 0.6,
            "pred_draw": 0.2,
            "pred_away": 0.2,
            "low": 0.4 if i == 0 else 0.7,
            "high": 0.6 if i == 0 else 0.3,
            "actual_home_score": int(actual.split("-")[0]),
            "actual_away_score": int(actual.split("-")[1]),
            "lambda_home": 2.5,
            "lambda_away": 1.5,
            "score1": actual if i == 0 else "2-1",
            "score2": "2-2",
            "score3": "3-3",
            "score4": "4-2",
        })
    frame = pd.DataFrame(rows)

    class Dummy:
        _competition_calibration_rows = {"NPB:npb_interleague:interleague": 80}
        _competition_temperatures = {"NPB:npb_interleague:interleague": 1.1}

    monkeypatch.setattr(bb, "RESULTS", tmp_path)
    Dummy.evaluate = bb.BaseballBacktest.evaluate
    Dummy.save_reports = bb.BaseballBacktest.save_reports
    Dummy.evaluate = lambda self, df, league: bb.BaseballBacktest.evaluate(self, df, league)
    bb.BaseballBacktest.save_reports(Dummy(), frame, "NPB")

    out = pd.read_csv(tmp_path / "npb_competition_target_metrics.csv")
    exact = out[out["Target"] == "exact_score"].iloc[0]
    hilo = out[out["Target"] == "low_high"].iloc[0]
    assert exact["Top1ExactScoreHitRate"] == 0.5
    assert exact["Top4ScoreHitRate"] == 1.0
    assert np.isfinite(float(hilo["LogLoss"]))
    assert np.isfinite(float(hilo["Brier"]))
    assert np.isfinite(float(hilo["ECE"]))



def test_external_data_discovery_manifest_is_unique_and_fail_closed():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "research" / "external_data_discovery.json").read_text(encoding="utf-8"))
    sources = payload["sources"]
    ids = [row["source_id"] for row in sources]
    assert len(ids) == len(set(ids))
    assert all(row["url"].startswith(("https://", "http://")) for row in sources)
    assert all(row["pit_status"] == "UNVERIFIED" for row in sources)
    assert all(row["research_status"] == "DISCOVERED" for row in sources)
