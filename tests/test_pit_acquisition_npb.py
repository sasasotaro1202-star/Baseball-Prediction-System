from datetime import datetime, timezone

from data.pit_acquisition import _parse_npb_schedule_html


def test_parse_npb_official_schedule_rows():
    html = """
    <table>
      <tr id="date0928">
        <td>9/28（月）</td>
        <td><div>DeNA</div> - <div>広島</div></td>
        <td><div>横　浜 18:00</div></td>
        <td></td>
        <td></td>
      </tr>
      <tr id="date0929">
        <td>9/29（火）</td>
        <td><div>巨人</div> - <div>広島</div></td>
        <td><div>東京ドーム 18:00</div></td>
        <td></td>
        <td></td>
      </tr>
    </table>
    """
    rows = _parse_npb_schedule_html(
        html,
        year=2026,
        month=9,
        start_date=datetime(2026, 9, 28, tzinfo=timezone.utc),
        end_date=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )

    assert len(rows) == 2
    assert rows[0]["home_team"] == "DeNA"
    assert rows[0]["away_team"] == "広島"
    assert rows[0]["event_date"] == "2026-09-28"
    assert rows[0]["start_time_local"] == "18:00"
    assert rows[0]["venue"] == "横 浜"
    assert rows[0]["game_id"].startswith("NPB-OFFICIAL-")


def test_parse_npb_schedule_rejects_reserve_days_and_wrong_month():
    html = """
    <table>
      <tr id="date0928">
        <td>9/28</td>
        <td>阪神 (予備日) - 中日</td>
        <td>甲子園 18:00</td>
        <td></td>
        <td></td>
      </tr>
      <tr id="date1001">
        <td>10/1</td>
        <td>巨人 - 阪神</td>
        <td>東京ドーム 18:00</td>
        <td></td>
        <td></td>
      </tr>
    </table>
    """
    rows = _parse_npb_schedule_html(html, year=2026, month=9)
    assert rows == []


def test_parse_npb_schedule_carries_forward_date_for_grouped_games():
    html = """
    <table>
      <tr id="date0928">
        <td>9/28（月）</td>
        <td>DeNA - 広島</td>
        <td>横　浜 18:00</td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td>西武 - 楽天</td>
        <td>ベルーナドーム 18:00</td>
        <td></td>
        <td></td>
      </tr>
    </table>
    """
    rows = _parse_npb_schedule_html(html, year=2026, month=9)

    assert len(rows) == 2
    assert rows[1]["event_date"] == "2026-09-28"
    assert rows[1]["home_team"] == "西武"
    assert rows[1]["away_team"] == "楽天"


def test_parse_npb_schedule_bounds_window_without_fabricating_starter_evidence():
    html = """
    <table>
      <tr id="date0927">
        <td>9/27</td>
        <td>巨人 - ヤクルト</td>
        <td>東京ドーム 18:00</td>
        <td></td>
        <td></td>
      </tr>
      <tr id="date0928">
        <td>9/28</td>
        <td>DeNA - 広島</td>
        <td>横　浜 18:00</td>
        <td></td>
        <td></td>
      </tr>
    </table>
    """
    rows = _parse_npb_schedule_html(
        html,
        year=2026,
        month=9,
        start_date=datetime(2026, 9, 28, tzinfo=timezone.utc),
        end_date=datetime(2026, 9, 28, tzinfo=timezone.utc),
    )

    assert len(rows) == 1
    assert rows[0]["event_date"] == "2026-09-28"
    assert "home_starter" not in rows[0]
    assert "away_starter" not in rows[0]


def test_parse_npb_schedule_supports_multiple_games_in_one_html_row():
    html = """
    <table>
      <tr id="date0928">
        <td>9/28（月）</td>
        <td>
          <div class="team1">DeNA</div> - <div class="team2">広島</div>
          <div class="team1">西武</div> - <div class="team2">楽天</div>
        </td>
        <td>
          <div class="place">横　浜 18:00</div>
          <div class="place">ベルーナドーム 18:00</div>
        </td>
      </tr>
    </table>
    """
    rows = _parse_npb_schedule_html(html, year=2026, month=9)

    assert len(rows) == 2
    assert rows[0]["home_team"] == "DeNA"
    assert rows[0]["away_team"] == "広島"
    assert rows[1]["home_team"] == "西武"
    assert rows[1]["away_team"] == "楽天"
    assert rows[0]["start_time_local"] == "18:00"
    assert rows[1]["start_time_local"] == "18:00"


def test_parse_npb_schedule_accepts_utf8_text():
    html = """
    <table>
      <tr id="date0928">
        <td>9/28</td>
        <td>阪神 - 中日</td>
        <td>甲子園 18:00</td>
      </tr>
    </table>
    """
    rows = _parse_npb_schedule_html(html.encode("utf-8").decode("utf-8"), year=2026, month=9)
    assert rows[0]["home_team"] == "阪神"
    assert rows[0]["away_team"] == "中日"

def test_build_npb_starter_pit_records_uses_observation_as_available_bound():
    from data.pit_acquisition import _build_npb_starter_pit_records

    rows = [{
        "home": "阪神タイガース",
        "away": "東京ヤクルトスワローズ",
        "home_starter": "先発A",
        "away_starter": "先発B",
        "official_start_time": "18:00",
        "starter_source": "https://npb.jp/announcement/starter/",
    }]
    schedule = [{
        "event_id": "NPB:official-game-1",
        "game_id": "NPB-OFFICIAL-GAME-1",
        "home_team": "阪神タイガース",
        "away_team": "東京ヤクルトスワローズ",
        "start_time_local": "18:00",
    }]

    observed = "2026-10-08T08:00:00+00:00"
    out = _build_npb_starter_pit_records(
        rows,
        target_date="2026-10-08",
        retrieved_at=observed,
        schedule_rows=schedule,
    )

    assert len(out) == 1
    assert out[0]["game_id"] == "NPB-OFFICIAL-GAME-1"
    assert out[0]["available_at"] == observed
    assert out[0]["home_starter_available_at"] == observed
    assert out[0]["away_starter_available_at"] == observed
    assert out[0]["home_starter_announced_at"] is None
    assert out[0]["away_starter_announced_at"] is None
    assert out[0]["observation_semantics"] == "available_at_is_observation_time; announcement_time_not_claimed"


def test_build_npb_starter_pit_records_rejects_incomplete_or_contradictory_rows():
    import pytest
    from data.pit_acquisition import _build_npb_starter_pit_records

    incomplete = [{
        "home": "阪神タイガース",
        "away": "東京ヤクルトスワローズ",
        "home_starter": "",
        "away_starter": "先発B",
        "official_start_time": "18:00",
    }]
    with pytest.raises(ValueError):
        _build_npb_starter_pit_records(
            incomplete,
            target_date="2026-10-08",
            retrieved_at="2026-10-08T08:00:00+00:00",
        )

    contradictory = [{
        "home": "阪神タイガース",
        "away": "東京ヤクルトスワローズ",
        "home_starter": "同一投手",
        "away_starter": "同一投手",
        "official_start_time": "18:00",
    }]
    with pytest.raises(ValueError):
        _build_npb_starter_pit_records(
            contradictory,
            target_date="2026-10-08",
            retrieved_at="2026-10-08T08:00:00+00:00",
        )

