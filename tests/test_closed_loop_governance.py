import json

from research.closed_loop_governance import calibration_stage, pit_revision_stage


def test_calibration_stage_accepts_current_per_league_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "leagues": {
                    "NPB": {"temperature": 0.925},
                    "MLB": {"temperature": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )

    stage = calibration_stage(path)

    assert stage.status == "READY"
    assert stage.blockers == ()


def test_calibration_stage_rejects_invalid_per_league_temperature(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "leagues": {
                    "NPB": {"temperature": 0.0},
                    "MLB": {"temperature": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )

    stage = calibration_stage(path)

    assert stage.status == "BLOCKED"
    assert stage.blockers == ("invalid_calibration:ValueError",)


def test_pit_revision_stage_blocks_historical_backfill(tmp_path):
    path = tmp_path / "source_snapshots.jsonl"
    rows = [
        {
            "league": "NPB",
            "entity_type": "game",
            "entity_id": "G1",
            "event_id": "G1",
            "source": "source-a",
            "payload_hash": "hash-v1",
            "available_at": "2026-09-10T12:00:00+00:00",
            "retrieved_at": "2026-09-10T12:01:00+00:00",
        },
        {
            "league": "NPB",
            "entity_type": "game",
            "entity_id": "G1",
            "event_id": "G1",
            "source": "source-a",
            "payload_hash": "hash-v2",
            "available_at": "2026-09-09T12:00:00+00:00",
            "retrieved_at": "2026-09-12T12:01:00+00:00",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    stage = pit_revision_stage(path)

    assert stage.status == "BLOCKED"
    assert stage.blockers == ("pit_backfill_detected:1",)


def test_holdout_stage_requires_both_leagues_and_untouched_flag(tmp_path):
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps({
        "NPB": {"used_for_candidate_selection": False},
        "MLB": {"used_for_candidate_selection": False},
    }), encoding="utf-8")
    from research.closed_loop_governance import holdout_stage
    assert holdout_stage(path).status == "READY"


def test_holdout_stage_blocks_selection_contamination(tmp_path):
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps({
        "NPB": {"used_for_candidate_selection": True},
        "MLB": {"used_for_candidate_selection": False},
    }), encoding="utf-8")
    from research.closed_loop_governance import holdout_stage
    stage = holdout_stage(path)
    assert stage.status == "BLOCKED"
    assert stage.blockers == ("holdout_selection_contamination:NPB",)


def test_candidate_stage_requires_both_leagues_and_explicit_decisions(tmp_path):
    path = tmp_path / "candidate_validation.json"
    path.write_text(json.dumps({
        "NPB": {
            "decision": "ADOPT",
            "baseline": {"LogLoss": 0.8, "Brier": 0.5, "Accuracy": 0.55},
            "candidate": {"LogLoss": 0.79, "Brier": 0.49, "Accuracy": 0.56},
        },
        "MLB": {
            "decision": "REJECT",
            "baseline": {"LogLoss": 0.72, "Brier": 0.52, "Accuracy": 0.47},
            "candidate": {"LogLoss": 0.70, "Brier": 0.50, "Accuracy": 0.47},
        },
    }), encoding="utf-8")
    from research.closed_loop_governance import candidate_stage
    assert candidate_stage(path).status == "READY"


def test_candidate_stage_blocks_incomplete_or_invalid_artifact(tmp_path):
    path = tmp_path / "candidate_validation.json"
    path.write_text(json.dumps({
        "NPB": {
            "decision": "MAYBE",
            "baseline": {"LogLoss": 0.8, "Brier": 0.5},
            "candidate": {"LogLoss": 0.79, "Brier": 0.49, "Accuracy": 0.56},
        },
    }), encoding="utf-8")
    from research.closed_loop_governance import candidate_stage
    stage = candidate_stage(path)
    assert stage.status == "BLOCKED"
    assert "missing_league:MLB" in stage.blockers
    assert "invalid_candidate_decision:NPB" in stage.blockers
    assert "invalid_candidate_metric:NPB:baseline:Accuracy" in stage.blockers
