import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from prediction.current_production import latest_target_date_jst


def test_latest_target_date_is_resolved_at_call_time_in_jst():
    expected = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    actual = latest_target_date_jst()
    assert re.fullmatch(r"20\\d{2}-\\d{2}-\\d{2}", actual)
    assert actual == expected
