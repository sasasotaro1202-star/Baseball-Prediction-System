from __future__ import annotations

from data import npb_team_player_context as ctx


BAT_HTML = """
<h3>個人打撃成績（セントラル・リーグ）</h3>
<table>
<tr><th>選手</th><th>試合</th><th>打席</th><th>打数</th><th>安打</th><th>二塁打</th><th>三塁打</th><th>本塁打</th><th>四球</th><th>死球</th><th>三振</th><th>盗塁</th><th>盗塁刺</th></tr>
<tr><td><a href="/bis/players/123.html">テスト　太郎</a></td><td>100</td><td>500</td><td>450</td><td>120</td><td>25</td><td>2</td><td>20</td><td>40</td><td>5</td><td>90</td><td>10</td><td>2</td></tr>
</table>
<p>2026年10月2日 現在</p>
"""

PIT_HTML = """
<h3>個人投手成績（セントラル・リーグ）</h3>
<table>
<tr><th>選手</th><th>登板</th><th>打者</th><th>投球回</th><th>安打</th><th>本塁打</th><th>四球</th><th>三振</th></tr>
<tr><td><a href="/bis/players/123.html">テスト　太郎</a></td><td>20</td><td>500</td><td>120</td><td>100</td><td>10</td><td>30</td><td>120</td></tr>
</table>
"""

DEF_HTML = """
<h4>捕手</h4>
<table>
<tr><th>選手</th><th>試合</th><th>失策</th><th>守備率</th></tr>
<tr><td><a href="/bis/players/123.html">テスト　太郎</a></td><td>50</td><td>2</td><td>.990</td></tr>
</table>
<h4>遊撃手</h4>
<table>
<tr><th>選手</th><th>試合</th><th>失策</th><th>守備率</th></tr>
<tr><td><a href="/bis/players/456.html">別選手</a></td><td>60</td><td>1</td><td>.995</td></tr>
</table>
"""


def test_parse_batting_and_player_id():
    rows, as_of = ctx.parse_stats_page(BAT_HTML, "batting")
    assert as_of == "2026-10-02"
    assert rows[0]["player_id"] == "123"
    assert rows[0]["player_name"] == "テスト 太郎"
    assert rows[0]["打席"] == "500"


def test_parse_fielding_preserves_position_sections():
    rows, _ = ctx.parse_stats_page(DEF_HTML, "fielding")
    assert rows[0]["position"] == "捕手"
    assert rows[1]["position"] == "遊撃手"


def test_derived_batting_metrics():
    rows, _ = ctx.parse_stats_page(BAT_HTML, "batting")
    derived = ctx._derive_batting(rows[0])
    assert derived["bb_pct"] == 0.08
    assert derived["k_pct"] == 0.18
    assert derived["hr_per_pa"] == 0.04
    assert derived["sb_attempt_rate"] == 0.024
    assert derived["bb_k_ratio"] == 40 / 90


def test_derived_pitching_metrics():
    rows, _ = ctx.parse_stats_page(PIT_HTML, "pitching")
    derived = ctx._derive_pitching(rows[0])
    assert derived["k9"] == 9.0
    assert derived["bb9"] == 2.25
    assert derived["hr9"] == 0.75
    assert derived["whip"] == (100 + 30) / 120
    assert derived["k_minus_bb_pct"] == (120 - 30) / 500


def test_merge_players_by_stable_id():
    pages = {
        "batting": {"rows": ctx.parse_stats_page(BAT_HTML, "batting")[0]},
        "pitching": {"rows": ctx.parse_stats_page(PIT_HTML, "pitching")[0]},
        "fielding": {"rows": ctx.parse_stats_page(DEF_HTML, "fielding")[0]},
    }
    players = ctx._merge_players("阪神タイガース", pages)
    first = next(x for x in players if x["player_id"] == "123")
    assert first["identity_status"] == "VERIFIED_STABLE_ID"
    assert first["batting"]["打席"] == "500"
    assert first["pitching"]["投球回"] == "120"
    assert first["fielding"][0]["position"] == "捕手"


def test_normalize_team():
    assert ctx.normalize_team("阪神") == "阪神タイガース"
    assert ctx.TEAM_SUFFIX["福岡ソフトバンクホークス"] == "h"

PROFILE_HTML = """
<table>
<tr><th>守備位置</th><td>投手</td><th>投打</th><td>右投右打</td></tr>
<tr><th>身長</th><td>190cm</td><th>体重</th><td>105kg</td></tr>
<tr><th>生年月日</th><td>1998年1月2日</td><th>経歴</th><td>テスト大学 - テスト球団</td></tr>
<tr><th>ドラフト</th><td>2019年1位</td></tr>
</table>
"""


def test_parse_profile_page_extracts_identity_and_physical_context():
    profile = ctx.parse_profile_page(PROFILE_HTML)
    assert profile["position"] == "投手"
    assert profile["handedness"] == "右投右打"
    assert profile["throws"] == "右"
    assert profile["bats"] == "右"
    assert profile["height_cm"] == 190.0
    assert profile["weight_kg"] == 105.0
    assert profile["birth_date"] == "1998年1月2日"
    assert profile["career"] == "テスト大学 - テスト球団"
    assert profile["draft"] == "2019年1位"


def test_derived_batting_totals_add_rate_metrics():
    rows, _ = ctx.parse_stats_page(BAT_HTML, "batting")
    derived = ctx._derive_batting(rows[0])
    assert round(derived["avg"], 6) == round(120 / 450, 6)
    assert round(derived["slg"], 6) == round((73 + 50 + 6 + 80) / 450, 6)
    assert "ops_from_totals" in derived
    assert derived["pa_per_game"] == 5.0
    assert derived["sb_success_rate"] == 10 / 12


def test_derived_fielding_metrics():
    rows, _ = ctx.parse_stats_page(DEF_HTML, "fielding")
    derived = ctx._derive_fielding(rows[0])
    assert derived["error_rate"] == 2 / 2
    assert derived["chances_per_game"] == 2 / 50
