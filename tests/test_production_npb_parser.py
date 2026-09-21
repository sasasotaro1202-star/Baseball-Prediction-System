from production_npb import parse_official_starters_html

def test_visible_team_name_duplication_falls_back_to_bounded_unit_extraction():
    html = """
    <h4>22月22日の予告先発投手</h4>
    <div class="unit">
      <img alt="読売ジャイアンツ">
      <div class="team_left"><span>読売ジャイアンツ</span></div>
      <div class="pitcher"><span>山田太郎</span></div>
    </div>
    <div class="unit">
      <img alt="阪神タイガース">
      <div class="team_left"><span>阪神タイガース</span></div>
      <div class="pitcher"><span>佐藤次郎</span></div>
    </div>
    <div class="game-time">18:00</div>
    """
    rows = parse_official_starters_html(html, "2026-02-22")
    assert rows == [{
        "home": "読売ジャイアンツ",
        "away": "阪神タイガース",
        "home_starter": "山田太郎",
        "away_starter": "佐藤次郎",
        "confirmed_starters": True,
        "starter_evidence_status": "official_announced",
        "starter_source": "https://npb.jp/announcement/starter/",
        "official_start_time": "18:00",
    }]
