from __future__ import annotations

from data import npb_roster_context as ctx


ROSTER_HTML = """
<h5>阪神タイガース</h5>
<ul>
  <li>投手</li><li>18</li>
  <li><a href="/bis/players/100.html">伊原　陵人</a></li>
  <li><a href="/bis/players/101.html">才木　浩人</a></li>
</ul>
<h5>横浜DeNAベイスターズ</h5>
<ul>
  <li>捕手</li><li>5</li>
  <li><a href="/bis/players/200.html">松尾　汐恩</a></li>
</ul>
"""


def test_parse_roster_page_groups_stable_player_ids_by_team():
    got = ctx.parse_roster_page(ROSTER_HTML, "2026-10-03")
    assert got["status"] == "AVAILABLE"
    assert got["target_date"] == "2026-10-03"
    assert got["player_count"] == 3
    assert [p["player_id"] for p in got["teams"]["阪神タイガース"]] == ["101", "100"] or {
        p["player_id"] for p in got["teams"]["阪神タイガース"]
    } == {"100", "101"}
    assert got["teams"]["横浜DeNAベイスターズ"][0]["player_id"] == "200"
    assert all(
        p["identity_status"] == "VERIFIED_STABLE_ID"
        for players in got["teams"].values()
        for p in players
    )


def test_parse_roster_page_deduplicates_player_ids():
    duplicated = ROSTER_HTML.replace(
        '<a href="/bis/players/101.html">才木　浩人</a>',
        '<a href="/bis/players/101.html">才木　浩人</a>'
        '<a href="/bis/players/101.html">才木　浩人</a>',
    )
    got = ctx.parse_roster_page(duplicated, "2026-10-03")
    assert got["player_count"] == 3


def test_collect_roster_context_is_fail_closed(monkeypatch):
    def fail_request(*args, **kwargs):
        raise RuntimeError("synthetic network failure")

    monkeypatch.setattr(ctx, "http_request", fail_request)
    got = ctx.collect_npb_roster_context("2026-10-03")
    assert got["status"] == "SOURCE_FAILED"
    assert got["teams"] == {}
    assert got["player_count"] == 0
    assert got["source"]["status"] == "SOURCE_FAILED"
    assert got["historical_oos_consumption"] == "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN"
