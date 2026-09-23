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


def test_predict_persists_blocked_state_for_impossible_starter_pair(monkeypatch, tmp_path):
    import production_npb

    monkeypatch.setattr(
        production_npb,
        "build_target_rows",
        lambda target_date: (_ for _ in ()).throw(
            RuntimeError(
                "PIT starter gate failed: identical starter assigned to both teams "
                "in one official game (A vs B): '投手A'."
            )
        ),
    )
    result = production_npb.predict("2026-09-24", str(tmp_path))
    assert result["execution_status"] == "BLOCKED_STARTERS"
    assert result["predictions"] == []
    assert "identical starter" in result["block_reason"]
    assert (tmp_path / "results" / "npb_production_2026-09-24.json").exists() is False


def _league_fixture(source_url: str, home: str, hs: str, away: str, as_: str) -> str:
    return (
        f'<h4>9月24日の予告先発</h4>'
        f'<div><span>18:00</span>'
        f'<img alt="{home}"><span>{hs}</span>'
        f'<img alt="{away}"><span>{as_}</span></div>'
        f'<h4>9月25日の予告先発</h4>'
    )


def test_official_league_starter_parser_supports_single_game_league_page():
    from production_npb import parse_official_league_starters_html

    cl = parse_official_league_starters_html(
        _league_fixture(
            "https://npb.jp/cl/",
            "広島東洋カープ", "森下　暢仁",
            "読売ジャイアンツ", "西舘　勇陽",
        ),
        "2026-09-24",
        "https://npb.jp/cl/",
    )
    assert cl == [{
        "home": "広島東洋カープ",
        "away": "読売ジャイアンツ",
        "home_starter": "森下　暢仁",
        "away_starter": "西舘　勇陽",
        "confirmed_starters": True,
        "starter_evidence_status": "official_announced",
        "starter_source": "https://npb.jp/cl/",
        "official_start_time": "18:00",
    }]


def test_official_starters_reconciles_suspect_dedicated_page_with_first_party_league_pages(monkeypatch):
    import production_npb

    cl = _league_fixture(
        "https://npb.jp/cl/",
        "広島東洋カープ", "森下　暢仁",
        "読売ジャイアンツ", "西舘　勇陽",
    )
    pl = _league_fixture(
        "https://npb.jp/pl/",
        "北海道日本ハムファイターズ", "達　孝太",
        "東北楽天ゴールデンイーグルス", "前田　健太",
    )

    monkeypatch.setattr(production_npb, "_load_official_starter_snapshot", lambda _: None)
    monkeypatch.setattr(
        production_npb,
        "parse_official_starters_html",
        lambda *_args, **_kwargs: [{
            "home": "広島東洋カープ", "away": "読売ジャイアンツ",
            "home_starter": "達　孝太", "away_starter": "達　孝太",
            "official_start_time": "18:00",
        }],
    )

    def fake_fetch(url):
        if "npb.jp/cl/" in url:
            return cl
        if "npb.jp/pl/" in url:
            return pl
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(production_npb, "fetch_text", fake_fetch)
    rows = production_npb.official_starters("2026-09-24")
    assert [(x["home"], x["home_starter"], x["away"], x["away_starter"]) for x in rows] == [
        ("広島東洋カープ", "森下　暢仁", "読売ジャイアンツ", "西舘　勇陽"),
        ("北海道日本ハムファイターズ", "達　孝太", "東北楽天ゴールデンイーグルス", "前田　健太"),
    ]
