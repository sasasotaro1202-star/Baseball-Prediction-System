from __future__ import annotations

import json

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
