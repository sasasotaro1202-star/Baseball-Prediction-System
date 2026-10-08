from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data import npb_pregame_context as ctx


def test_parse_official_games_extracts_venue_and_start_time():
    html = """
    <html><body><table>
      <tr><td>阪神</td><td>甲子園</td><td>18:00</td><td>巨人</td></tr>
      <tr><td>福岡ソフトバンクホークス</td><td>みずほPayPayドーム</td><td>14:00</td><td>埼玉西武ライオンズ</td></tr>
    </table></body></html>
    """
    rows = ctx.parse_official_games(html, "2026-10-04")
    assert rows[0]["home"] == "阪神タイガース"
    assert rows[0]["away"] == "読売ジャイアンツ"
    assert rows[0]["venue"] == "阪神甲子園球場"
    assert rows[0]["official_start_time"] == "18:00"
    assert rows[1]["venue"] == "みずほPayPayドーム福岡"


def test_parse_standings_preserves_team_record_and_source_time():
    html = """
    <table>
      <tr><th>Team</th><th>G</th><th>W</th><th>L</th><th>T</th><th>PCT</th><th>GB</th><th>Home</th><th>Road</th></tr>
      <tr><td>Hanshin Tigers</td><td>138</td><td>77</td><td>59</td><td>2</td><td>.566</td><td>-</td><td>35-32</td><td>42-27</td></tr>
    </table>
    """
    got = ctx.parse_standings(html, "https://example.test/std_c.html", "2026-10-04T00:00:00+00:00")
    row = got["阪神タイガース"]
    assert row["games"] == 138
    assert row["wins"] == 77
    assert row["losses"] == 59
    assert row["draws"] == 2
    assert row["win_pct"] == 0.566
    assert row["home_record"] == "35-32"
    assert row["road_record"] == "42-27"
    assert row["published_or_observed_at_utc"] == "2026-10-04T00:00:00+00:00"


def test_fetch_weather_maps_target_hour_without_postgame_fields(monkeypatch):
    payload = {
        "hourly": {
            "time": ["2026-10-04T18:00", "2026-10-04T19:00"],
            "temperature_2m": [22, 21], "apparent_temperature": [22, 21],
            "relative_humidity_2m": [65, 67], "dew_point_2m": [15, 15],
            "precipitation_probability": [10, 20], "precipitation": [0.0, 0.1],
            "rain": [0.0, 0.1], "wind_speed_10m": [8, 9], "wind_gusts_10m": [14, 15],
            "wind_direction_10m": [250, 250], "pressure_msl": [1011, 1010],
            "cloud_cover": [40, 45], "weather_code": [1, 2],
        }
    }
    class Resp:
        def json(self):
            return payload
    monkeypatch.setattr(ctx, "http_request", lambda *args, **kwargs: Resp())
    got = ctx.fetch_weather("阪神甲子園球場", pd.Timestamp("2026-10-04T09:00:00Z"))
    assert got["status"] == "AVAILABLE"
    assert got["temperature_c"] == 22.0
    assert got["precipitation_probability_pct"] == 10.0
    assert got["wind_gust_kmh"] == 14.0
    assert got["available_at_utc"] is not None


def test_weather_unknown_venue_fails_closed(monkeypatch):
    got = ctx.fetch_weather("UNKNOWN-VENUE", pd.Timestamp("2026-10-04T09:00:00Z"))
    assert got["status"] == "UNAVAILABLE_VENUE_UNKNOWN"
    assert got["temperature_c"] is None


def test_collect_context_records_pit_boundary_and_snapshot(monkeypatch):
    game_html = """
    <table><tr><td>阪神</td><td>甲子園</td><td>18:00</td><td>巨人</td></tr></table>
    """
    standing_html = """
    <table>
      <tr><th>Team</th><th>G</th><th>W</th><th>L</th><th>T</th><th>PCT</th><th>GB</th><th>Home</th><th>Road</th></tr>
      <tr><td>Hanshin Tigers</td><td>138</td><td>77</td><td>59</td><td>2</td><td>.566</td><td>-</td><td>35-32</td><td>42-27</td></tr>
      <tr><td>Yomiuri Giants</td><td>142</td><td>76</td><td>63</td><td>3</td><td>.547</td><td>2.5</td><td>39-30</td><td>37-33</td></tr>
    </table>
    """
    def fake_fetch(url, timeout=(8, 45)):
        if "games/gm" in url:
            return game_html, "2026-10-04T00:00:01+00:00"
        return standing_html, "2026-10-04T00:00:02+00:00"
    def fake_weather(venue, start):
        return {
            "status": "AVAILABLE", "source": "open_meteo_forecast",
            "available_at_utc": "2026-10-04T00:00:03+00:00",
            "requested_time_jst": start.tz_convert("Asia/Tokyo").isoformat(),
            "temperature_c": 22.0,
        }
    monkeypatch.setattr(ctx, "_fetch_text", fake_fetch)
    monkeypatch.setattr(ctx, "fetch_weather", fake_weather)
    snapshot = ctx.collect_npb_pregame_context("2026-10-04", now_utc=pd.Timestamp("2026-10-04T00:00:00Z"))
    assert snapshot["prediction_cutoff_utc"] == "2026-10-04T00:00:00+00:00"
    assert snapshot["status"] == "AVAILABLE"
    assert snapshot["historical_oos_consumption"] == "DISABLED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN"
    assert snapshot["game_count"] == 1
    assert snapshot["games"][0]["standing_home"]["wins"] == 77
    assert snapshot["games"][0]["weather"]["temperature_c"] == 22.0
    assert len(snapshot["snapshot_id"]) == 64
    assert snapshot["sources"][0]["source_id"] == "npb_official_game_schedule_context"


def test_snapshot_id_is_deterministic_for_identical_content():
    base = {"a": 1, "b": [2, 3]}
    assert ctx._snapshot_id(base) == ctx._snapshot_id(json.loads(json.dumps(base)))


def test_schedule_source_uses_official_japanese_endpoint():
    assert ctx.NPB_DAY_URL == "https://npb.jp/bis/{year}/games/gm{date}.html"


def test_parse_realistic_japanese_schedule_row():
    html = """
    <table>
      <tr>
        <th>対戦カード</th><th>球場・開始時間</th><th>予告先発</th>
      </tr>
      <tr>
        <td>ヤクルト</td><td>神宮 18:00</td><td>吉村</td>
        <td>広島</td>
      </tr>
      <tr>
        <td>DeNA</td><td>横浜 18:00</td><td>平良</td>
        <td>阪神</td>
      </tr>
    </table>
    """
    rows = ctx.parse_official_games(html, "2026-10-04")
    assert len(rows) == 2
    assert rows[0]["home"] in {"東京ヤクルトスワローズ", "横浜DeNAベイスターズ"}
    assert rows[0]["official_start_time"] == "18:00"
    assert rows[0]["venue"] in {"明治神宮野球場", "横浜スタジアム"}

def test_parse_official_schedule_detail_extracts_compound_matchups():
    html = """
    <table>
      <tr><td>10/4（日）</td><td>ヤクルト - 広島</td><td>神宮</td><td>18:00</td></tr>
      <tr><td>10/4（日）</td><td>DeNA － 阪神</td><td>横浜</td><td>18:00</td></tr>
      <tr><td>10/3（土）</td><td>巨人 2 - 5 DeNA</td><td>東京ドーム</td><td>18:00</td></tr>
    </table>
    """
    rows = ctx.parse_official_schedule_detail(html, "2026-10-04")
    assert len(rows) == 2
    assert rows[0]["home"] == "東京ヤクルトスワローズ"
    assert rows[0]["away"] == "広島東洋カープ"
    assert rows[0]["venue"] == "明治神宮野球場"
    assert rows[1]["home"] == "横浜DeNAベイスターズ"
    assert rows[1]["away"] == "阪神タイガース"


def test_collect_context_falls_back_to_monthly_official_schedule(monkeypatch):
    daily_url_fragment = "games/gm"
    monthly_html = """
    <table><tr><td>10/4（日）</td><td>ヤクルト - 広島</td><td>神宮</td><td>18:00</td></tr></table>
    """
    standing_html = """
    <table>
      <tr><th>Team</th><th>G</th><th>W</th><th>L</th><th>T</th><th>PCT</th></tr>
      <tr><td>Hanshin Tigers</td><td>1</td><td>1</td><td>0</td><td>0</td><td>.1000</td></tr>
    </table>
    """
    def fake_fetch(url, timeout=(8, 45)):
        if daily_url_fragment in url:
            raise RuntimeError("daily endpoint failure")
        if "schedule_10_detail.html" in url:
            return monthly_html, "2026-10-04T00:00:01+00:00"
        return standing_html, "2026-10-04T00:00:02+00:00"

    def fake_weather(venue, start):
        return {"status": "AVAILABLE", "available_at_utc": "2026-10-04T00:00:03+00:00", "temperature_c": 22.0}

    monkeypatch.setattr(ctx, "_fetch_text", fake_fetch)
    monkeypatch.setattr(ctx, "fetch_weather", fake_weather)
    got = ctx.collect_npb_pregame_context("2026-10-04", now_utc=pd.Timestamp("2026-10-04T00:00:00Z"))
    assert got["game_count"] == 1
    schedule = next(x for x in got["sources"] if x["source_id"] == "npb_official_game_schedule_context")
    assert schedule["endpoint_variant"] == "MONTH_DETAIL_FALLBACK"
    assert "daily endpoint" in schedule["primary_daily_endpoint_error"]


def test_parse_official_schedule_detail_accepts_full_canonical_team_names():
    html = """
    <table><tr>
      <td>10/4（日）</td>
      <td>東京ヤクルトスワローズ - 広島東洋カープ</td>
      <td>明治神宮野球場</td>
      <td>18:00</td>
    </tr></table>
    """
    rows = ctx.parse_official_schedule_detail(html, "2026-10-04")
    assert len(rows) == 1
    assert rows[0]["home"] == "東京ヤクルトスワローズ"
    assert rows[0]["away"] == "広島東洋カープ"


def test_parse_official_games_accepts_image_alt_and_embedded_time():
    html = """
    <table>
      <tr>
        <td><img src="yakult.png" alt="ヤクルト"></td>
        <td><img src="jingu.png" alt="神宮"> 18:00</td>
        <td><a title="広島">広島</a></td>
      </tr>
    </table>
    """
    rows = ctx.parse_official_games(html, "2026-10-04")
    assert len(rows) == 1
    assert rows[0]["home"] == "東京ヤクルトスワローズ"
    assert rows[0]["away"] == "広島東洋カープ"
    assert rows[0]["venue"] == "明治神宮野球場"
    assert rows[0]["official_start_time"] == "18:00"


def test_parse_official_schedule_detail_accepts_day_only_date_cell():
    html = """
    <table>
      <tr><td>4</td><td>ヤクルト</td><td>神宮</td><td>18:00</td><td>広島</td></tr>
      <tr><td>3</td><td>巨人</td><td>東京ドーム</td><td>18:00</td><td>DeNA</td></tr>
    </table>
    """
    rows = ctx.parse_official_schedule_detail(html, "2026-10-04")
    assert len(rows) == 1
    assert rows[0]["home"] == "東京ヤクルトスワローズ"
    assert rows[0]["away"] == "広島東洋カープ"


def test_parse_official_games_falls_back_to_non_table_game_blocks():
    html = """
    <div class="game"><img alt="ヤクルト"><span>神宮</span><span>18:00</span><img alt="広島"></div>
    <div class="game"><img alt="DeNA"><span>横浜</span><span>18:00</span><img alt="阪神"></div>
    <div class="game"><img alt="ロッテ"><span>ZOZOマリン</span><span>18:00</span><img alt="楽天"></div>
    """
    rows = ctx.parse_official_games(html, "2026-10-04")
    assert len(rows) == 3
    assert [(r["home"], r["away"]) for r in rows] == [
        ("東京ヤクルトスワローズ", "広島東洋カープ"),
        ("横浜DeNAベイスターズ", "阪神タイガース"),
        ("千葉ロッテマリーンズ", "東北楽天ゴールデンイーグルス"),
    ]
    assert [r["venue"] for r in rows] == [
        "明治神宮野球場", "横浜スタジアム", "ZOZOマリンスタジアム"
    ]


def test_parse_monthly_schedule_falls_back_to_non_table_game_blocks():
    html = """
    <div>10/4（日）</div>
    <div>ヤクルト</div><div>神宮</div><div>18:00</div><div>広島</div>
    <div>DeNA</div><div>横浜</div><div>18:00</div><div>阪神</div>
    """
    rows = ctx.parse_official_schedule_detail(html, "2026-10-04")
    assert len(rows) == 2
    assert rows[0]["home"] == "東京ヤクルトスワローズ"
    assert rows[0]["away"] == "広島東洋カープ"

def test_pregame_workflow_does_not_escape_jst_date_expression():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "baseball_pregame_context.yml"
    ).read_text(encoding="utf-8")
    assert '--date "\\${{ github.event.inputs.date || steps.date.outputs.date }}"' not in workflow
    assert '--date "${{ github.event.inputs.date || steps.date.outputs.date }}"' in workflow
