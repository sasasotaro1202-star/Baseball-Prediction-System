from argparse import Namespace

import run as run_module
import prediction.current_production as current_production


def _args():
    return Namespace(league="MLB", date="2026-09-24", data_dir="data")


def _blocked_runtime(status):
    return {"execution_status": status, "predictions": []}


def test_run_predict_propagates_missing_runtime_as_failure(monkeypatch):
    monkeypatch.setattr(current_production, "predict_current", lambda **_: _blocked_runtime("BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME"))
    assert run_module.cmd_predict(_args()) == 1


def test_run_predict_keeps_expected_starter_block_retryable(monkeypatch):
    monkeypatch.setattr(current_production, "predict_current", lambda **_: _blocked_runtime("BLOCKED_STARTERS"))
    assert run_module.cmd_predict(_args()) == 0


def test_direct_production_cli_propagates_missing_runtime(monkeypatch, capsys):
    monkeypatch.setattr(current_production, "predict_current", lambda **_: _blocked_runtime("BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME"))
    assert current_production.main(["--league", "MLB", "--date", "2026-09-24"]) == 1
    capsys.readouterr()


def test_direct_production_cli_keeps_starter_block_retryable(monkeypatch, capsys):
    monkeypatch.setattr(current_production, "predict_current", lambda **_: _blocked_runtime("BLOCKED_STARTERS"))
    assert current_production.main(["--league", "NPB", "--date", "2026-09-24"]) == 0
    capsys.readouterr()


def test_production_dispatcher_keeps_diagnostics_off_stdout(monkeypatch, capsys):
    import json
    import sys
    import types

    monkeypatch.setattr(
        current_production,
        "current_runtime",
        lambda league: {
            "available": True,
            "league": league,
            "status": "AVAILABLE",
            "entrypoint": "production_npb",
            "model_version": "test-model",
            "contract": "TEST_CONTRACT",
        },
    )

    module = types.ModuleType("production_npb")
    def fake_predict(*args, **kwargs):
        print("[PREDICTOR DIAGNOSTIC] captured")
        return {"execution_status": "EXECUTED", "predictions": []}
    module.predict = fake_predict
    monkeypatch.setitem(sys.modules, "production_npb", module)

    assert current_production.main(["--league", "NPB"]) == 0
    captured = capsys.readouterr()
    assert "[PREDICTOR DIAGNOSTIC] captured" not in captured.out
    assert "[PREDICTOR DIAGNOSTIC] captured" in captured.err
    payload = json.loads(captured.out)
    assert payload["execution_status"] == "EXECUTED"
    assert payload["current_production_runtime"]["model_version"] == "test-model"
