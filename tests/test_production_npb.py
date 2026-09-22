from pathlib import Path
from production_npb import build_target_rows, parse_official_starters_html

TEAMS = [
    ("読売ジャイアンツ","小笠原　慎之介"),("東京ヤクルトスワローズ","高橋　奎二"),
    ("中日ドラゴンズ","大野　雄大"),("広島東洋カープ","森　翔平"),
    ("阪神タイガース","才木　浩人"),("横浜DeNAベイスターズ","竹田　祐"),
    ("北海道日本ハムファイターズ","有原　航平"),("オリックス・バファローズ","髙島　泰都"),
    ("東北楽天ゴールデンイーグルス","瀧中　瞭太"),("福岡ソフトバンクホークス","大津　亮介"),
    ("千葉ロッテマリーンズ","Ａ．ジャクソン"),("埼玉西武ライオンズ","武内　夏暉"),
]

def fixture_html():
    pairs=[]
    for i in range(0,12,2):
        h,hs=TEAMS[i]; a,ass=TEAMS[i+1]
        pairs.append(f'<a href="/team"><img alt="{h}"></a><a href="/pitcher">{hs}</a>'
                     f'<a href="/team"><img alt="{a}"></a><a href="/pitcher">{ass}</a>'
                     f'<span>（球場）{"14:00" if i in (0,6,8) else "18:00"}</span>')
    return '<h4>9月20日の予告先発投手</h4>' + ''.join(pairs)

def test_parser_extracts_six_official_games_without_network():
    d=parse_official_starters_html(fixture_html(),"2026-09-20")
    assert len(d)==6
    assert all(x["confirmed_starters"] for x in d)
    assert all(x["starter_evidence_status"]=="official_announced" for x in d)
    assert [x["official_start_time"] for x in d]==["14:00","18:00","18:00","14:00","14:00","18:00"]

def test_20260920_has_six_pit_safe_games(monkeypatch):
    import production_npb as p
    import pandas as pd
    monkeypatch.setattr(p, "fetch_text", lambda url: fixture_html())
    # The production builder intentionally filters games that have already
    # started. Freeze the clock before the first fixture game so this
    # historical fixture tests the PIT/starter contract rather than time gating.
    monkeypatch.setattr(
        p, "_utc_now", lambda: pd.Timestamp("2026-09-20 03:00:00+00:00")
    )
    d=build_target_rows("2026-09-20")
    assert len(d)==6
    assert d["confirmed_starters"].all()
    assert (d["starter_evidence_status"]=="official_announced").all()
    assert d["home_score"].isna().all()
    assert d["away_score"].isna().all()


def fixture_html_20260921_five_games():
    games = [
        ("中日ドラゴンズ","髙橋　宏斗","広島東洋カープ","斉藤　優汰","14:00"),
        ("阪神タイガース","Ｅ．ルーカス","横浜DeNAベイスターズ","平良　拳太郎","14:00"),
        ("北海道日本ハムファイターズ","加藤　貴之","オリックス・バファローズ","山口　廉王","14:00"),
        ("東北楽天ゴールデンイーグルス","前田　健太","福岡ソフトバンクホークス","上茶谷　大河","13:00"),
        ("千葉ロッテマリーンズ","Ａ．ジャクソン","埼玉西武ライオンズ","武内　夏暉","18:00"),
    ]
    parts = []
    for h, hs, a, ass, tm in games:
        parts.append(
            f'<div class="unit"><img alt="{h}"><span>{hs}</span>'
            f'<img alt="{a}"><span>{ass}</span><span>（球場）{tm}</span></div>'
        )
    return '<h4>9月21日の予告先発投手</h4>' + ''.join(parts)


def test_parser_extracts_five_official_games_and_exact_starters():
    d = parse_official_starters_html(fixture_html_20260921_five_games(), "2026-09-21")
    assert len(d) == 5
    assert [(x["home"], x["away"], x["home_starter"], x["away_starter"], x["official_start_time"]) for x in d] == [
        ("中日ドラゴンズ","広島東洋カープ","髙橋 宏斗","斉藤 優汰","14:00"),
        ("阪神タイガース","横浜DeNAベイスターズ","Ｅ．ルーカス","平良 拳太郎","14:00"),
        ("北海道日本ハムファイターズ","オリックス・バファローズ","加藤 貴之","山口 廉王","14:00"),
        ("東北楽天ゴールデンイーグルス","福岡ソフトバンクホークス","前田 健太","上茶谷 大河","13:00"),
        ("千葉ロッテマリーンズ","埼玉西武ライオンズ","Ａ．ジャクソン","武内 夏暉","18:00"),
    ]


def test_recovery_regime_diagnostics_are_recomputed_per_game():
    source = Path("production_npb.py").read_text(encoding="utf-8")
    assert "recovery_regime_label = str(bt._regime_router.labels(xrow)[0])" in source
    assert '"regime":recovery_regime_label' in source
    assert '"classification_regime_model_weights":recovery_regime_weights' in source


def test_target_rows_reject_non_official_starter_source(monkeypatch):
    import production_npb as p
    import pandas as pd

    monkeypatch.setattr(
        p,
        "official_starters",
        lambda target_date: [{
            "home": "読売ジャイアンツ",
            "away": "阪神タイガース",
            "home_starter": "投手A",
            "away_starter": "投手B",
            "confirmed_starters": True,
            "starter_evidence_status": "official_announced",
            "starter_source": "https://example.com/not-official",
            "official_start_time": "18:00",
        }],
    )
    monkeypatch.setattr(
        p,
        "_utc_now",
        lambda: pd.Timestamp("2026-09-20 00:00:00+00:00"),
    )
    try:
        p.build_target_rows("2026-09-20")
    except RuntimeError as exc:
        assert "allowlisted official NPB" in str(exc)
    else:
        raise AssertionError("non-official starter source was accepted")
