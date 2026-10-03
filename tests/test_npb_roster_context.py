from __future__ import annotations

from data import npb_roster_context as ctx


ROSTER_HTML = """
<h4>セントラル・リーグ</h4>
<h5>出場選手登録</h5>
<table><tr><td>阪神タイガース</td><td>投手</td><td>18</td><td><a href="/bis/players/100.html">伊原　陵人</a></td></tr></table>
<h5>出場選手登録抹消</h5>
<table><tr><td>広島東洋カープ</td><td>投手</td><td>50</td><td><a href="/bis/players/300.html">杉田　健</a></td></tr></table>
<h5>出場選手一覧</h5>
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
    assert {p["player_id"] for p in got["teams"]["阪神タイガース"]} == {"100", "101"}
    assert all(p["identity_status"] == "VERIFIED_STABLE_ID" for p in got["teams"]["阪神タイガース"])
    assert got["teams"]["横浜DeNAベイスターズ"][0]["player_id"] == "200"
    assert all(
        p["identity_status"] == "VERIFIED_STABLE_ID"
        for players in got["teams"].values()
        for p in players
    )



def test_parse_roster_page_captures_transactions_and_player_state():
    got = ctx.parse_roster_page(ROSTER_HTML, "2026-10-03")
    assert got["registered_today_count"] == 1
    assert got["removed_today_count"] == 1
    assert got["registered_today"][0]["player_id"] == "100"
    assert got["registered_today"][0]["transaction_status"] == "REGISTERED"
    assert got["registered_today"][0]["position"] == "投手"
    assert got["registered_today"][0]["uniform_number"] == "18"
    assert got["removed_today"][0]["player_id"] == "300"
    assert got["removed_today"][0]["transaction_status"] == "REMOVED"
    hanshin_100 = next(
        p for p in got["teams"]["阪神タイガース"] if p["player_id"] == "100"
    )
    assert hanshin_100["roster_transaction_status"] == "REGISTERED_TODAY"


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


ROSTER_TABLE_HTML = """
<h5>出場選手一覧</h5>
<h5>阪神タイガース</h5>
<table>
<tr><td>投手</td><td>13</td><td>岩崎　優</td></tr>
<tr><td>捕手</td><td>2</td><td>梅野　隆太郎</td></tr>
<tr><td>内野手</td><td>3</td><td>大山　悠輔</td></tr>
</table>
"""


def test_parse_roster_page_supports_official_plain_text_roster_table_rows():
    got = ctx.parse_roster_page(ROSTER_TABLE_HTML, "2026-10-03")
    assert got["player_count"] == 3
    assert [p["player_name"] for p in got["teams"]["阪神タイガース"]] == ["大山 悠輔", "岩崎 優", "梅野 隆太郎"]
    assert {p["uniform_number"] for p in got["teams"]["阪神タイガース"]} == {"13", "2", "3"}
    assert all(p["identity_status"] == "NAME_ONLY_UNVERIFIED" for p in got["teams"]["阪神タイガース"])
    assert all(p["roster_transaction_status"] == "IDENTITY_UNVERIFIED" for p in got["teams"]["阪神タイガース"])


PLAYER_SEARCH_HTML = """
<a href="/bis/players/91495138.html">32 外野手 濱田　太貴 阪神タイガース</a>
<a href="/bis/players/99999999.html">濱田　太貴 読売ジャイアンツ</a>
"""


def test_parse_player_search_results_extracts_official_stable_ids():
    rows = ctx._parse_player_search_results(PLAYER_SEARCH_HTML)
    assert rows[0]["player_id"] == "91495138"
    assert rows[0]["player_url"].endswith("/bis/players/91495138.html")


def test_resolve_roster_player_ids_uses_exact_name_and_team(monkeypatch):
    snapshot = {
        "schema_version": "npb-roster-context-v1",
        "target_date": "2026-10-03",
        "teams": {
            "阪神タイガース": [
                {
                    "player_id": None,
                    "player_name": "濱田　太貴",
                    "player_url": None,
                    "identity_status": "NAME_ONLY_UNVERIFIED",
                }
            ]
        },
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }
    seen_urls = []

    def fake_fetch(url):
        seen_urls.append(url)
        return PLAYER_SEARCH_HTML, "2026-10-03T00:00:00+00:00"

    monkeypatch.setattr(ctx, "_fetch_text", fake_fetch)
    got = ctx.resolve_roster_player_ids(snapshot, teams={"阪神タイガース"})
    player = got["teams"]["阪神タイガース"][0]
    assert player["player_id"] == "91495138"
    assert player["identity_status"] == "VERIFIED_STABLE_ID"
    assert player["identity_resolution_status"] == "RESOLVED_EXACT_OFFICIAL_PLAYER_SEARCH"
    assert got["identity_resolution"]["resolved_count"] == 1
    assert got["identity_resolution"]["resolved_rate"] == 1.0
    assert len(seen_urls) == 1


def test_resolve_roster_player_ids_keeps_ambiguous_identity_unverified(monkeypatch):
    html = """
    <a href="/bis/players/11111111.html">濱田　太貴 阪神タイガース</a>
    <a href="/bis/players/22222222.html">濱田　太貴 阪神タイガース</a>
    """
    snapshot = {
        "teams": {
            "阪神タイガース": [
                {"player_id": None, "player_name": "濱田　太貴", "player_url": None}
            ]
        },
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
    }
    monkeypatch.setattr(ctx, "_fetch_text", lambda url: (html, "2026-10-03T00:00:00+00:00"))
    got = ctx.resolve_roster_player_ids(snapshot, teams={"阪神タイガース"})
    player = got["teams"]["阪神タイガース"][0]
    assert player["player_id"] is None
    assert player["identity_resolution_status"] == "IDENTITY_AMBIGUOUS"
    assert got["identity_resolution"]["ambiguous_count"] == 1
