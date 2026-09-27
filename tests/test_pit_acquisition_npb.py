from datetime import datetime, timezone

from data.pit_acquisition import _parse_npb_schedule_html


def test_parse_npb_official_schedule_rows():
    html = """
    <table>
      <tr id="date0928">
        <td class="date">9/28（月）</td>
        <td><div class="team1"><a>DeNA</a></div> - <div class="team2"><a>広島</a></div></td>
        <td><div class="place">横　浜 18:00</div></td>
      </tr>
      <tr id="date0929">
        <td class="date">9/29（火）</td>
        <td><div class="team1">巨人</div> - <div class="team2">広島</div></td>
        <td><div class="place">東京ドーム 18:00</div></td>
      </tr>
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
        <td><div class="team1">阪神 (予備日)</div> - <div class="team2">中日</div></td>
        <td><div class="place">甲子園 18:00</div></td>
      </tr>
      <tr id="date1001">
        <td>10/1</td>
        <td><div class="team1">巨人</div> - <div class="team2">阪神</div></td>
        <td><div class="place">東京ドーム 18:00</div></td>
      </tr>
    """
    rows = _parse_npb_schedule_html(html, year=2026, month=9)
    assert rows == []


def test_parse_npb_schedule_bounds_window_without_fabricating_starter_evidence():
    html = """
    <tr id="date0927">
      <td>9/27</td>
      <td><div class="team1">巨人</div> - <div class="team2">ヤクルト</div></td>
      <td><div class="place">東京ドーム 18:00</div></td>
    </tr>
    <tr id="date0928">
      <td>9/28</td>
      <td><div class="team1">DeNA</div> - <div class="team2">広島</div></td>
      <td><div class="place">横　浜 18:00</div></td>
    </tr>
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
