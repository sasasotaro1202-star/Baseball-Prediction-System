from production_npb import parse_official_starters_html

def test_visible_team_name_duplication_falls_back_to_bounded_unit_extraction():
    html = """
    <h4>2月22日の予告先発投手</h4>
    <div class="unit">
      <img alt="読売ジャイアンツ">
      <span>読売ジャイアンツ</span>
      <div class="team_left"><span>山田太郎</span></div>
    </div>
    <div class="unit">
      <img alt="阪神タイガース">
      <span>阪神タイガース</span>
      <div class="team_left"><span>佐藤次郎</span></div>
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


def test_competition_metadata_accepts_explicit_official_regular_season_schedule():
    from production_npb import _official_daily_competition_metadata

    html = """
    <html>
      <body>
        <h1>2026 NPB Games</h1>
        <h2>10月6日（火）</h2>
        <div>阪神タイガース 18:00 広島東洋カープ</div>
      </body>
    </html>
    """

    import production_npb

    original = production_npb.fetch_text

    def fake_fetch(url):
        assert url == "https://npb.jp/games/2026/schedule.html"
        return "<h2>2026年度 セ・パ公式戦</h2><div>10月6日（火）</div>"

    production_npb.fetch_text = fake_fetch
    try:
        label = _official_daily_competition_metadata(
            html,
            "https://npb.jp/bis/eng/2026/games/gm20261006.html",
            "2026-10-06",
        )
    finally:
        production_npb.fetch_text = original

    assert label["status"] == "classified"
    assert label["competition"] == "npb_regular"
    assert label["stage"] == "regular_season"
    assert label["competition_key"] == "NPB:npb_regular:regular_season"
    assert label["source_field"] == "npb_annual_schedule_heading"
    assert label["source_value"] == "2026年度 セ・パ公式戦"


def test_starter_page_clock_regex_accepts_real_clock():
    import re

    assert re.fullmatch(r"\d{1,2}:\d{2}", "18:00")
