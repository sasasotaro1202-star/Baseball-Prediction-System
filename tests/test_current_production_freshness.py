import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from prediction.current_production import latest_target_date_jst


def test_latest_target_date_is_resolved_at_call_time_in_jst():
    expected = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    actual = latest_target_date_jst()
    assert re.fullmatch(r"20\d{2}-\d{2}-\d{2}", actual)
    assert actual == expected

def test_predict_current_rejects_non_live_target_date(monkeypatch):
    import prediction.current_production as cp

    monkeypatch.setattr(cp, "current_runtime", lambda league: {
        "available": True,
        "league": league,
        "status": "AVAILABLE",
        "entrypoint": "unsupported-for-test",
        "model_version": "test",
        "contract": "test",
        "formal_adoption_status": "CURRENT_PRODUCTION",
    })
    monkeypatch.setattr(cp, "latest_target_date_jst", lambda: "2026-09-29")
    result = cp.predict_current(league="NPB", target_date="2026-09-28")
    assert result["execution_status"] == "BLOCKED_STALE_TARGET_DATE"
    assert result["requested_target_date"] == "2026-09-28"
    assert result["resolved_target_date"] == "2026-09-29"



def test_predict_current_forwards_pregame_only_to_registered_runtime(monkeypatch):
    import production_npb
    import prediction.current_production as cp

    captured = {}

    monkeypatch.setattr(cp, "current_runtime", lambda league: {
        "available": True,
        "league": league,
        "status": "AVAILABLE",
        "entrypoint": "production_npb",
        "model_version": "test-model",
        "contract": "test-contract",
        "formal_adoption_status": "CURRENT_PRODUCTION",
    })
    monkeypatch.setattr(cp, "latest_target_date_jst", lambda: "2026-10-01")

    def fake_predict(target_date, data_dir, *, pregame_only=False):
        captured["target_date"] = target_date
        captured["data_dir"] = data_dir
        captured["pregame_only"] = pregame_only
        return {"execution_status": "NO_DUE_PREGAME_GAMES", "predictions": []}

    monkeypatch.setattr(production_npb, "predict", fake_predict)

    result = cp.predict_current(
        league="NPB",
        target_date="2026-10-01",
        data_dir="data",
        pregame_only=True,
    )
    assert result["execution_status"] == "NO_DUE_PREGAME_GAMES"
    assert captured == {
        "target_date": "2026-10-01",
        "data_dir": "data",
        "pregame_only": True,
    }



def test_blocked_runtime_preserves_on_demand_request_provenance(monkeypatch):
    import prediction.current_production as cp

    monkeypatch.setattr(cp, "latest_target_date_jst", lambda: "2026-10-03")
    monkeypatch.setattr(cp, "current_runtime", lambda league: {
        "available": False,
        "league": league,
        "status": "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME",
        "reason": "blocked for test",
    })

    result = cp.predict_current(league="NPB")
    assert result["execution_status"] == "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME"
    assert result["prediction_request_mode"] == "on_demand"
    assert result["prediction_schedule"] == "on_demand"
    assert result["prediction_source"] == "MANUAL_LIVE"
    assert result["resolved_target_date"] == "2026-10-03"
    assert result["prediction_requested_at_utc"] == result["prediction_generated_at"]



def test_git_commit_prefers_checked_out_head_over_dispatch_sha(monkeypatch):
    import prediction.current_production as cp

    monkeypatch.setenv("GITHUB_SHA", "dispatch-sha")
    monkeypatch.setenv("BASEBALL_CHECKED_OUT_SHA", "checked-out-sha")
    assert cp._git_commit() == "checked-out-sha"


def test_git_commit_falls_back_to_dispatch_sha(monkeypatch):
    import prediction.current_production as cp

    monkeypatch.delenv("BASEBALL_CHECKED_OUT_SHA", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "dispatch-sha")
    assert cp._git_commit() == "dispatch-sha"
