from production_npb import parse_official_starters_html


def test_official_starter_parser_handles_five_game_slate_in_page_order():
    games = [
        ("中日ドラゴンズ", "髙橋 宏斗", "広島東洋カープ", "斉藤 優汰", "14:00"),
        ("阪神タイガース", "Ｅ．ルーカス", "横浜DeNAベイスターズ", "平良 拳太郎", "14:00"),
        ("北海道日本ハムファイターズ", "加藤 貴之", "オリックス・バファローズ", "山口 廉王", "14:00"),
        ("東北楽天ゴールデンイーグルス", "前田 健太", "福岡ソフトバンクホークス", "上茶谷 大河", "13:00"),
        ("千葉ロッテマリーンズ", "Ａ．ジャクソン", "埼玉西武ライオンズ", "武内 夏暉", "18:00"),
    ]
    parts = ["<h4>9月21日の予告先発投手</h4>"]
    for home, hp, away, ap, tm in games:
        parts.append(
            f'<div class="unit"><span>{tm}</span>'
            f'<img alt="{home}"><div class="team_left"><span>{hp}</span></div>'
            f'<img alt="{away}"><div class="team_left"><span>{ap}</span></div></div>'
        )
    parts.append("<h4>9月22日の予告先発投手</h4>")
    parsed = parse_official_starters_html("".join(parts), "2026-09-21")
    assert len(parsed) == 5
    assert [(g["home"], g["home_starter"], g["away"], g["away_starter"], g["official_start_time"]) for g in parsed] == games


def test_official_starter_parser_fails_when_game_time_is_missing():
    html = (
        "<h4>9月21日の予告先発投手</h4>"
        '<div class="unit"><img alt="中日ドラゴンズ"><div class="team_left"><span>髙橋 宏斗</span></div>'
        '<img alt="広島東洋カープ"><div class="team_left"><span>斉藤 優汰</span></div></div>'
        "<h4>9月22日の予告先発投手</h4>"
    )
    try:
        parse_official_starters_html(html, "2026-09-21")
    except RuntimeError as exc:
        assert "no official game times" in str(exc)
    else:
        raise AssertionError("missing official game time must fail closed")


def test_official_starter_parser_rejects_identical_starters_in_one_game():
    html = (
        "<h4>9月21日の予告先発投手</h4>"
        '<div class="unit"><span>18:00</span>'
        '<img alt="広島東洋カープ"><div class="team_left"><span>投手A</span></div>'
        '<img alt="読売ジャイアンツ"><div class="team_left"><span>投手A</span></div></div>'
        "<h4>9月22日の予告先発投手</h4>"
    )
    try:
        parse_official_starters_html(html, "2026-09-21")
    except RuntimeError as exc:
        assert "identical starter" in str(exc)
    else:
        raise AssertionError("identical starters for both teams must fail closed")
