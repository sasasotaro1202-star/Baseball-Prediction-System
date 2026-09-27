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
