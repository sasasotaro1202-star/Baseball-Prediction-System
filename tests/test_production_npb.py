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
    monkeypatch.setattr(p, "fetch_text", lambda url: fixture_html())
    d=build_target_rows("2026-09-20")
    assert len(d)==6
    assert d["confirmed_starters"].all()
    assert (d["starter_evidence_status"]=="official_announced").all()
    assert d["home_score"].isna().all()
    assert d["away_score"].isna().all()
