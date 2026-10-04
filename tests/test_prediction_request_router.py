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
                    "predictions": [{"game_id": "g1", "home": "H", "away": "A"}]}),
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
            "predictions": [{"game_id": "g1", "home": "H", "away": "A"}],
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
