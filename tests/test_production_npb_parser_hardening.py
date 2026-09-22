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



def test_official_starter_parser_does_not_discard_team_bounded_evidence_when_units_are_incomplete():
    html = (
        "<h4>9月22日の予告先発投手</h4>"
        # Six structurally complete units are insufficient for the six-game
        # target, while six additional teams are still recoverable by the
        # team-bounded extractor.
        '<div class="unit"><img alt="東京ヤクルトスワローズ"><div class="team_left"><span>松本 健吾</span></div></div>'
        '<div class="unit"><img alt="阪神タイガース"><div class="team_left"><span>伊藤 将司</span></div></div>'
        '<div class="unit"><img alt="横浜DeNAベイスターズ"><div class="team_left"><span>東 克樹</span></div></div>'
        '<div class="unit"><img alt="中日ドラゴンズ"><div class="team_left"><span>Ｋ．マラー</span></div></div>'
        '<div class="unit"><img alt="広島東洋カープ"><div class="team_left"><span>玉村 昇悟</span></div></div>'
        '<div class="unit"><img alt="読売ジャイアンツ"><div class="team_left"><span>戸郷 翔征</span></div></div>'
        # The remaining six team/starter records use a presentation variant
        # that the structural parser intentionally ignores but the bounded
        # team extractor can still resolve.
        '<div class="unit"><img alt="北海道日本ハムファイターズ"><p>北山 亘基</p></div>'
        '<div class="unit"><img alt="東北楽天ゴールデンイーグルス"><p>伊藤 樹</p></div>'
        '<div class="unit"><img alt="千葉ロッテマリーンズ"><p>Ａ．ジャクソン</p></div>'
        '<div class="unit"><img alt="オリックス・バファローズ"><p>東松 快征</p></div>'
        '<div class="unit"><img alt="福岡ソフトバンクホークス"><p>上茶谷 大河</p></div>'
        '<div class="unit"><img alt="埼玉西武ライオンズ"><p>佐藤 爽</p></div>'
        "<div>18:00</div><div>18:00</div><div>15:00</div>"
        "<div>18:00</div><div>18:00</div><div>18:00</div>"
    )
    parsed = parse_official_starters_html(html, "2026-09-22")
    assert len(parsed) == 6
    assert [(g["home"], g["home_starter"], g["away"], g["away_starter"]) for g in parsed] == [
        ("東京ヤクルトスワローズ", "松本 健吾", "阪神タイガース", "伊藤 将司"),
        ("横浜DeNAベイスターズ", "東 克樹", "中日ドラゴンズ", "Ｋ．マラー"),
        ("広島東洋カープ", "玉村 昇悟", "読売ジャイアンツ", "戸郷 翔征"),
        ("北海道日本ハムファイターズ", "北山 亘基", "東北楽天ゴールデンイーグルス", "伊藤 樹"),
        ("千葉ロッテマリーンズ", "Ａ．ジャクソン", "オリックス・バファローズ", "東松 快征"),
        ("福岡ソフトバンクホークス", "上茶谷 大河", "埼玉西武ライオンズ", "佐藤 爽"),
    ]
