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
