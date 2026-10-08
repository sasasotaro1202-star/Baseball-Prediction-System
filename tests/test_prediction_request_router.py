from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.prediction_request as router


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_route_prefers_current_production(monkeypatch, tmp_path):
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r1",
        "competition_id": "MLB",
        "target_date": "today",
    }
    write_json(tmp_path / "request.json", request)
    policy = {
        "result_directory": "prediction_requests/results",
        "max_runtime_seconds": 10,
        "production_commands": {"MLB": ["python", "prod.py", "{target_date}"]},
        "validated_research_shadow_commands": {"MLB": ["python", "shadow.py", "{target_date}"]},
    }
    runtime = {
        "runtimes": {
            "MLB": {
                "formal_adoption_status": "CURRENT_PRODUCTION",
                "entrypoint": "mlb_prod",
                "model_version": "m1",
                "contract": "MLB_HOME_AWAY_V1",
            }
        }
    }
    monkeypatch.setattr(router, "POLICY_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(router, "RUNTIME_PATH", tmp_path / "runtime.json")
    monkeypatch.setattr(router, "_today_jst", lambda: "2026-10-04")
    monkeypatch.setattr(router, "_run", lambda command, timeout_seconds: (
        0,
        json.dumps({"execution_status": "EXECUTED", "pit_status": "PASS",
                    "predictions": [{"game_id": "g1", "home": "H", "away": "A", "competition": "npb_regular", "competition_stage": "regular_season", "competition_classification_status": "classified", "competition_key": "NPB:npb_regular:regular_season"}]}),
        "",
    ))
    write_json(tmp_path / "policy.json", policy)
    write_json(tmp_path / "runtime.json", runtime)

    monkeypatch.chdir(tmp_path)
    rc = router.main(["--request-file", str(tmp_path / "request.json")])
    assert rc == 0
    result = json.loads((tmp_path / "prediction_requests/results/r1.json").read_text(encoding="utf-8"))
    assert result["generation_lane"] == "CURRENT_PRODUCTION_RUNTIME"
    assert result["generation_status"] == "EXECUTED"


def test_route_uses_validated_research_shadow_when_production_is_unavailable(monkeypatch, tmp_path):
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r2",
        "competition_id": "NPB",
        "target_date": "today",
    }
    write_json(tmp_path / "request.json", request)
    write_json(tmp_path / "policy.json", {
        "result_directory": "prediction_requests/results",
        "max_runtime_seconds": 10,
        "production_commands": {"NPB": ["python", "prod.py", "{target_date}"]},
        "validated_research_shadow_commands": {"NPB": ["python", "shadow.py", "{target_date}"]},
    })
    write_json(tmp_path / "runtime.json", {
        "runtimes": {
            "NPB": {
                "formal_adoption_status": "BLOCKED_UNTIL_ADOPTED",
                "entrypoint": "production_npb",
                "model_version": "m1",
                "contract": "NPB_HOME_DRAW_AWAY_V1",
            }
        }
    })
    monkeypatch.setattr(router, "POLICY_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(router, "RUNTIME_PATH", tmp_path / "runtime.json")
    monkeypatch.setattr(router, "_today_jst", lambda: "2026-10-04")
    calls = []

    def fake_run(command, timeout_seconds):
        calls.append(command)
        generated = tmp_path / "results/npb_shadow_2026-10-04.json"
        generated.parent.mkdir(parents=True, exist_ok=True)
        write_json(generated, {
            "execution_status": "RESEARCH_SHADOW_EXECUTED",
            "pit_status": "PASS",
            "scope": "RESEARCH_SHADOW",
            "production_eligibility": False,
            "predictions": [{"game_id": "g1", "home": "H", "away": "A", "competition": "npb_regular", "competition_stage": "regular_season", "competition_classification_status": "classified", "competition_key": "NPB:npb_regular:regular_season"}],
        })
        return 0, "", ""

    monkeypatch.setattr(router, "_run", fake_run)
    monkeypatch.chdir(tmp_path)
    rc = router.main(["--request-file", str(tmp_path / "request.json")])
    assert rc == 0
    assert calls[0][0:2] == ["python", "shadow.py"]
    result = json.loads((tmp_path / "prediction_requests/results/r2.json").read_text(encoding="utf-8"))
    assert result["generation_lane"] == "VALIDATED_RESEARCH_SHADOW"
    assert result["generation_status"] == "RESEARCH_SHADOW_EXECUTED"


def test_unverifiable_generation_status_is_preserved():
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r-unverifiable",
        "competition_id": "MLB",
        "target_date": "2026-10-04",
    }
    payload = {
        "execution_status": "GENERATION_OUTPUT_UNVERIFIABLE",
        "pit_status": "UNKNOWN",
        "predictions": [],
    }
    router._validate_generated_output(payload, request, "CURRENT_PRODUCTION_RUNTIME")


def test_generation_failure_status_is_preserved():
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r-failed",
        "competition_id": "MLB",
        "target_date": "2026-10-04",
    }
    payload = {
        "execution_status": "GENERATION_FAILED",
        "pit_status": "UNKNOWN",
        "predictions": [],
    }
    router._validate_generated_output(payload, request, "CURRENT_PRODUCTION_RUNTIME")


def test_route_uses_mlb_competition_specific_research_runtime_when_production_missing(monkeypatch, tmp_path):
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "mlb-research-route",
        "competition_id": "MLB",
        "target_date": "today",
    }
    write_json(tmp_path / "request.json", request)
    write_json(tmp_path / "policy.json", {
        "result_directory": "prediction_requests/results",
        "max_runtime_seconds": 10,
        "production_commands": {"MLB": ["python", "production.py", "{target_date}"]},
        "validated_research_shadow_commands": {},
        "competition_specific_research_commands": {
            "MLB": ["python", "-m", "prediction.mlb_research_preview", "--date", "{target_date}"]
        },
        "output_paths": {
            "MLB": {
                "COMPETITION_SPECIFIC_RESEARCH_RUNTIME":
                    "results/mlb_research_preview_{target_date}.json"
            }
        },
    })
    write_json(tmp_path / "runtime.json", {"runtimes": {"MLB": {
        "formal_adoption_status": "BLOCKED_UNTIL_REGISTERED",
        "entrypoint": "",
        "model_version": "",
        "contract": "",
    }}})
    monkeypatch.setattr(router, "POLICY_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(router, "RUNTIME_PATH", tmp_path / "runtime.json")
    monkeypatch.setattr(router, "_today_jst", lambda: "2026-10-08")

    calls = []
    def fake_run(command, timeout_seconds):
        calls.append(command)
        generated = tmp_path / "results/mlb_research_preview_2026-10-08.json"
        write_json(generated, {
            "execution_status": "RESEARCH_SHADOW_EXECUTED",
            "scope": "RESEARCH_SHADOW",
            "production_eligibility": False,
            "pit_status": "UNVERIFIABLE",
            "predictions": [{
                "game_id": "g1", "home": "H", "away": "A",
                "competition": "mlb_regular",
                "competition_stage": "regular_season",
                "competition_key": "MLB:mlb_regular:regular_season",
                "competition_classification_status": "classified",
            }],
        })
        return 0, "", ""

    monkeypatch.setattr(router, "_run", fake_run)
    monkeypatch.chdir(tmp_path)
    rc = router.main(["--request-file", str(tmp_path / "request.json")])

    assert rc == 0
    assert calls == [[
        "python", "-m", "prediction.mlb_research_preview",
        "--date", "2026-10-08"
    ]]
    result = json.loads(
        (tmp_path / "prediction_requests/results/mlb-research-route.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["generation_lane"] == "COMPETITION_SPECIFIC_RESEARCH_RUNTIME"
    assert result["generation_status"] == "RESEARCH_SHADOW_EXECUTED"


def test_unknown_competition_fails_closed(monkeypatch, tmp_path):
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r3",
        "competition_id": "UNKNOWN_COMP",
        "target_date": "today",
    }
    write_json(tmp_path / "request.json", request)
    write_json(tmp_path / "policy.json", {
        "result_directory": "prediction_requests/results",
        "max_runtime_seconds": 10,
        "production_commands": {},
        "validated_research_shadow_commands": {},
    })
    write_json(tmp_path / "runtime.json", {"runtimes": {}})
    monkeypatch.setattr(router, "POLICY_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(router, "RUNTIME_PATH", tmp_path / "runtime.json")
    monkeypatch.setattr(router, "_today_jst", lambda: "2026-10-04")
    monkeypatch.chdir(tmp_path)
    rc = router.main(["--request-file", str(tmp_path / "request.json")])
    assert rc == 0
    result = json.loads((tmp_path / "prediction_requests/results/r3.json").read_text(encoding="utf-8"))
    assert result["generation_lane"] == "BLOCKED"
    assert result["generation_status"] == "BLOCKED_NO_VALIDATED_PREDICTION_RUNTIME"
    assert not result["prediction_output"]["predictions"]


def test_past_date_is_rejected(monkeypatch, tmp_path):
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r4",
        "competition_id": "NPB",
        "target_date": "2026-10-03",
    }
    write_json(tmp_path / "request.json", request)
    monkeypatch.setattr(router, "_today_jst", lambda: "2026-10-04")
    with pytest.raises(ValueError, match="past target dates"):
        router._resolve_target_date(request)


def test_research_shadow_accepts_unverifiable_pit_when_marked_nonproduction():
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r-unverifiable-research",
        "competition_id": "NPB",
        "target_date": "2026-10-04",
    }
    payload = {
        "execution_status": "RESEARCH_SHADOW_EXECUTED",
        "scope": "RESEARCH_SHADOW",
        "production_eligibility": False,
        "pit_status": "UNVERIFIABLE",
        "predictions": [{"game_id": "g1", "home": "H", "away": "A", "competition": "npb_regular", "competition_stage": "regular_season", "competition_classification_status": "classified", "competition_key": "NPB:npb_regular:regular_season"}],
    }
    router._validate_generated_output(payload, request, "VALIDATED_RESEARCH_SHADOW")


def test_current_production_rejects_unverifiable_pit():
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r-unverifiable-production",
        "competition_id": "NPB",
        "target_date": "2026-10-04",
    }
    payload = {
        "execution_status": "EXECUTED",
        "pit_status": "UNVERIFIABLE",
        "predictions": [{"game_id": "g1", "home": "H", "away": "A", "competition": "npb_regular", "competition_stage": "regular_season", "competition_classification_status": "classified", "competition_key": "NPB:npb_regular:regular_season"}],
    }
    with pytest.raises(ValueError, match="current production prediction must have PIT PASS"):
        router._validate_generated_output(payload, request, "CURRENT_PRODUCTION_RUNTIME")


def test_production_lane_falls_back_to_validated_research_shadow(monkeypatch, tmp_path):
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r-prod-fallback",
        "competition_id": "NPB",
        "target_date": "today",
    }
    write_json(tmp_path / "request.json", request)
    write_json(tmp_path / "policy.json", {
        "result_directory": "prediction_requests/results",
        "max_runtime_seconds": 10,
        "production_commands": {"NPB": ["python", "prod.py", "{target_date}"]},
        "validated_research_shadow_commands": {"NPB": ["python", "shadow.py", "{target_date}"]},
    })
    write_json(tmp_path / "runtime.json", {
        "runtimes": {
            "NPB": {
                "formal_adoption_status": "CURRENT_PRODUCTION",
                "entrypoint": "production_npb",
                "model_version": "prod-v1",
                "contract": "NPB_HOME_DRAW_AWAY_V1",
            }
        }
    })
    monkeypatch.setattr(router, "POLICY_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(router, "RUNTIME_PATH", tmp_path / "runtime.json")
    monkeypatch.setattr(router, "_today_jst", lambda: "2026-10-05")

    calls = []

    def fake_run(command, timeout_seconds):
        calls.append(command)
        if command[1] == "prod.py":
            generated = tmp_path / "results/npb_production_2026-10-05.json"
            generated.parent.mkdir(parents=True, exist_ok=True)
            write_json(generated, {
                "execution_status": "BLOCKED_STARTERS",
                "pit_status": "NOT_RUN",
                "predictions": [],
            })
            return 0, "", "starter data unavailable"
        generated = tmp_path / "results/npb_shadow_2026-10-05.json"
        generated.parent.mkdir(parents=True, exist_ok=True)
        write_json(generated, {
            "execution_status": "RESEARCH_SHADOW_EXECUTED",
            "scope": "RESEARCH_SHADOW",
            "production_eligibility": False,
            "pit_status": "UNVERIFIABLE",
            "predictions": [{"game_id": "g1", "home": "H", "away": "A", "competition": "npb_regular", "competition_stage": "regular_season", "competition_classification_status": "classified", "competition_key": "NPB:npb_regular:regular_season"}],
        })
        return 0, "", ""

    monkeypatch.setattr(router, "_run", fake_run)
    monkeypatch.chdir(tmp_path)
    rc = router.main(["--request-file", str(tmp_path / "request.json")])

    assert rc == 0
    assert len(calls) == 2
    assert calls[0][0:2] == ["python", "prod.py"]
    assert calls[1][0:2] == ["python", "shadow.py"]
    result = json.loads(
        (tmp_path / "prediction_requests/results/r-prod-fallback.json").read_text(encoding="utf-8")
    )
    assert result["generation_lane"] == "VALIDATED_RESEARCH_SHADOW"
    assert result["generation_status"] == "RESEARCH_SHADOW_EXECUTED"
    assert result["prediction_output"]["user_fallback_from_lane"] == "CURRENT_PRODUCTION_RUNTIME"
    assert result["prediction_output"]["primary_production_status"] == "BLOCKED_STARTERS"


def test_research_prediction_rejects_unknown_competition_classification():
    request = {
        "schema_version": "baseball-prediction-request-v1",
        "request_id": "r-unknown-competition",
        "competition_id": "NPB",
        "target_date": "2026-10-06",
    }
    payload = {
        "execution_status": "RESEARCH_SHADOW_EXECUTED",
        "scope": "RESEARCH_SHADOW",
        "production_eligibility": False,
        "pit_status": "PASS",
        "predictions": [{
            "game_id": "g1",
            "home": "H",
            "away": "A",
            "competition": "npb_unknown",
            "competition_stage": "unknown",
            "competition_classification_status": "unknown",
        }],
    }
    with pytest.raises(ValueError, match="competition classification must be explicit"):
        router._validate_generated_output(payload, request, "VALIDATED_RESEARCH_SHADOW")


def test_repository_policy_registers_mlb_research_preview_without_production_promotion():
    policy = json.loads(
        (Path(__file__).resolve().parents[1] / "config" / "prediction_request_policy.json")
        .read_text(encoding="utf-8")
    )
    command = policy["competition_specific_research_commands"]["MLB"]
    assert command[:3] == ["python", "-m", "prediction.mlb_research_preview"]
    assert policy["output_paths"]["MLB"]["COMPETITION_SPECIFIC_RESEARCH_RUNTIME"].startswith(
        "results/mlb_research_preview_"
    )


def test_mlb_research_preview_contract_is_explicitly_nonproduction():
    from prediction.mlb_research_preview import OUTPUT_SCHEMA, MODEL_VERSION

    assert OUTPUT_SCHEMA == "baseball-mlb-research-preview-v1"
    assert MODEL_VERSION == "mlb-research-preview-ensemble-v1"




def test_mlb_preview_schedule_rows_include_backtest_league_key(monkeypatch):
    from prediction.mlb_research_preview import _schedule
    import pandas as pd

    class FakeBacktest:
        def _get_json(self, url, params):
            return {
                "dates": [{
                    "date": "2026-10-09",
                    "games": [{
                        "gamePk": 123,
                        "gameDate": "2026-10-09T05:00:00Z",
                        "status": {"abstractGameState": "Preview"},
                        "gameType": "R",
                        "seriesDescription": "Regular Season",
                        "season": "2026",
                        "teams": {
                            "home": {"team": {"name": "Home Team"}},
                            "away": {"team": {"name": "Away Team"}},
                        },
                        "venue": {"name": "Test Park"},
                    }],
                }]
            }

    rows, retrieved = _schedule(
        FakeBacktest(),
        pd.Timestamp("2026-10-09", tz="Asia/Tokyo").date(),
        pd.Timestamp("2026-10-08T13:00:00Z").to_pydatetime(),
    )
    assert rows
    assert rows[0]["league"] == "MLB"
    assert rows[0]["competition_classification_status"] == "classified"
    assert retrieved
