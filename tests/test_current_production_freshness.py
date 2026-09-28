import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from prediction.current_production import latest_target_date_jst


def test_latest_target_date_is_resolved_at_call_time_in_jst():
    expected = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    actual = latest_target_date_jst()
    assert re.fullmatch(r"20\\d{2}-\\d{2}-\\d{2}", actual)
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
