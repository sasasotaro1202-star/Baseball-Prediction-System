import json

from research.closed_loop_governance import calibration_stage


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
