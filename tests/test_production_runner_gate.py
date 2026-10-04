import pytest

from data.availability import AvailabilityRecord
from prediction.runner import eligibility_gate, run_production_prediction


def availability(league="NPB"):
    return AvailabilityRecord(
        event_id="g1",
        league=league,
        home_team="HOME",
        away_team="AWAY",
        home_starter="P1",
        away_starter="P2",
        home_starter_announced_at="2026-01-01T00:00:00+00:00",
        away_starter_announced_at="2026-01-01T00:00:00+00:00",
        lineup_status="UNVERIFIABLE",
        lineup_announced_at=None,
        source="https://npb.jp/",
        retrieved_at="2026-01-01T00:05:00+00:00",
        prediction_cutoff="2026-01-01T00:05:00+00:00",
        event_start_at="2026-01-01T01:00:00+00:00",
    )


def test_research_gate_can_validate_registered_wbc():
    ok, reasons = eligibility_gate(
        availability=availability("WBC"),
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
    )
    assert ok
    assert reasons == []


def test_production_gate_rejects_research_only_wbc():
    ok, reasons = eligibility_gate(
        availability=availability("WBC"),
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
        production=True,
    )
    assert not ok
    assert "competition_not_production_eligible" in reasons


def test_production_gate_rejects_pit_safe_npb_until_adoption():
    ok, reasons = eligibility_gate(
        availability=availability("NPB"),
        required_data_ok=True,
        feature_complete=True,
        model_available=True,
        calibration_available=True,
        production=True,
    )
    assert not ok
    assert reasons == ["competition_not_production_eligible"]


def test_production_gate_fails_closed_for_unknown_competition():
    with pytest.raises(ValueError, match="unknown competition_id/league"):
        eligibility_gate(
            availability=availability("UNKNOWN"),
            required_data_ok=True,
            feature_complete=True,
            model_available=True,
            calibration_available=True,
            production=True,
        )


def test_explicit_production_entry_point_cannot_fall_back_to_research_mode():
    future = "2999-01-01T00:00:00+00:00"
    # The wrapper must force production=True even if a caller tries to pass
    # production=False. NPB is currently fail-closed, so the forced production
    # gate must reject the prediction before any filesystem side effect occurs.
    result = run_production_prediction(
            row={
                "required_data_ok": True,
                "feature_complete": True,
                "model_available": True,
                "calibration_available": True,
            },
            availability=AvailabilityRecord(
                event_id="g2",
                league="NPB",
                home_team="HOME",
                away_team="AWAY",
                home_starter="P1",
                away_starter="P2",
                home_starter_announced_at="2998-12-31T23:00:00+00:00",
                away_starter_announced_at="2998-12-31T23:00:00+00:00",
                lineup_status="UNVERIFIABLE",
                lineup_announced_at=None,
                source="https://npb.jp/",
                retrieved_at=future,
                prediction_cutoff=future,
                event_start_at="2999-01-01T01:00:00+00:00",
            ),
            probability_fn=lambda row: {"home": 0.5, "draw": 0.2, "away": 0.3},
            model_version="test",
            feature_version="test",
            calibration_version="test",
            git_commit="test",
            data_snapshot_id="test",
            log_path="/tmp/unused.jsonl",
            production=False,
        )
    assert result["eligible"] is False
    assert result["reasons"] == ["competition_not_production_eligible"]


def test_current_production_runtime_registry_is_stable():
    from prediction.current_production import current_runtime

    npb = current_runtime("NPB")
    assert npb["available"] is False
    assert npb["status"] == "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME"
    assert npb["formal_adoption_status"] == "BLOCKED_UNTIL_ADOPTED"

    mlb = current_runtime("MLB")
    assert mlb["available"] is False
    assert mlb["status"] == "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME"


def test_current_production_never_uses_research_fallback(monkeypatch):
    import prediction.current_production as cp

    monkeypatch.setattr(cp, "CONFIG", cp.ROOT / "config" / "current_production_runtime.json")
    result = cp.predict_current(league="MLB", target_date="2026-09-24")
    assert result["execution_status"] == "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME"
    assert result["predictions"] == []


def test_run_prediction_preserves_full_competition_metadata(monkeypatch):
    import prediction.runner as runner

    captured = []
    monkeypatch.setattr(runner, "append_prediction", lambda record, path: captured.append(record))

    result = runner.run_prediction(
        row={
            "required_data_ok": True,
            "feature_complete": True,
            "model_available": True,
            "calibration_available": True,
            "competition_key": "NPB:npb_regular:regular_season",
            "competition_stage": "regular_season",
            "season_type": "regular_season",
            "game_class": "official",
            "competition_classification_status": "classified",
            "competition_metadata_source": "https://npb.jp/bis/eng/2026/games/gm20261004.html",
            "competition_metadata_source_field": "npb_daily_schedule_heading",
            "competition_metadata_source_value": "公式戦【試合予定】",
        },
        availability=availability("NPB"),
        probability_fn=lambda row: {"home": 0.5, "draw": 0.2, "away": 0.3},
        model_version="test-model",
        feature_version="test-features",
        calibration_version="test-calibration",
        git_commit="test-commit",
        data_snapshot_id="test-snapshot",
        log_path="/tmp/unused.jsonl",
    )

    assert result["eligible"] is True
    assert len(captured) == 1
    record = captured[0]
    assert record.competition_key == "NPB:npb_regular:regular_season"
    assert record.competition_stage == "regular_season"
    assert record.season_type == "regular_season"
    assert record.game_class == "official"
    assert record.competition_classification_status == "classified"
    assert record.competition_metadata_source == "https://npb.jp/bis/eng/2026/games/gm20261004.html"
    assert record.competition_metadata_source_field == "npb_daily_schedule_heading"
    assert record.competition_metadata_source_value == "公式戦【試合予定】"
