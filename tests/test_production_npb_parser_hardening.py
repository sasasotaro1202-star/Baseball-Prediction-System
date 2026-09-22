from production_npb import parse_official_starters_html


def test_official_starter_parser_rejects_duplicate_team_tokens():
    html = (
        "<h4>9月21日の予告先発投手</h4>"
        '<div class="unit"><span>14:00</span>'
        '<img alt="中日ドラゴンズ"><div class="team_left"><span>髙橋 宏斗</span></div>'
        '<img alt="広島東洋カープ"><div class="team_left"><span>斉藤 優汰</span></div></div>'
        '<div class="unit"><span>14:00</span>'
        '<img alt="中日ドラゴンズ"><div class="team_left"><span>別投手</span></div>'
        '<img alt="阪神タイガース"><div class="team_left"><span>投手</span></div></div>'
        "<h4>9月22日の予告先発投手</h4>"
    )
    try:
        parse_official_starters_html(html, "2026-09-21")
    except RuntimeError as exc:
        assert "duplicate team tokens" in str(exc)
    else:
        raise AssertionError("duplicate team tokens must fail closed")


def test_official_starter_parser_does_not_discard_team_bounded_evidence_when_units_are_per_team():
    html = (
        "<h4>9月22日の予告先発投手</h4>"
        '<div class="unit"><img alt="東京ヤクルトスワローズ"><div class="team_left"><span>松本 健吾</span></div></div>'
        '<div class="unit"><img alt="阪神タイガース"><div class="team_left"><span>伊藤 将司</span></div></div>'
        '<div class="unit"><img alt="横浜DeNAベイスターズ"><div class="team_left"><span>東 克樹</span></div></div>'
        '<div class="unit"><img alt="中日ドラゴンズ"><div class="team_left"><span>Ｋ．マラー</span></div></div>'
        "<div>18:00</div><div>18:00</div>"
    )
    parsed = parse_official_starters_html(html, "2026-09-22")
    assert len(parsed) == 2
    assert [(g["home"], g["home_starter"], g["away"], g["away_starter"]) for g in parsed] == [
        ("東京ヤクルトスワローズ", "松本 健吾", "阪神タイガース", "伊藤 将司"),
        ("横浜DeNAベイスターズ", "東 克樹", "中日ドラゴンズ", "Ｋ．マラー"),
    ]
