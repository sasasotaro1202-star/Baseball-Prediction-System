from __future__ import annotations

from data import npb_player_context as ctx


def test_derive_batter_advanced_metrics():
    row = {
        "打席_num": 500.0, "打数_num": 450.0, "安打_num": 120.0,
        "本塁打_num": 20.0, "四球_num": 40.0, "死球_num": 5.0,
        "三振_num": 90.0, "盗塁_num": 10.0, "盗塁刺_num": 2.0,
        "二塁打_num": 25.0, "三塁打_num": 2.0,
    }
    out = ctx._derive_advanced(row, "batting")
    assert out["bb_pct"] == 0.08
    assert out["k_pct"] == 0.18
    assert out["hr_per_pa"] == 0.04
    assert out["sb_attempt_rate"] == 0.024
    assert out["bb_k_ratio"] == 40 / 90


def test_derive_pitcher_advanced_metrics():
    row = {
        "投球回_num": 120.0, "三振_num": 120.0, "四球_num": 30.0,
        "本塁打_num": 10.0, "安打_num": 100.0, "死球_num": 4.0,
        "打者_num": 500.0,
    }
    out = ctx._derive_advanced(row, "pitching")
    assert out["k9"] == 9.0
    assert out["bb9"] == 2.25
    assert out["hr9"] == 0.75
    assert out["whip"] == (100 + 30) / 120
    assert out["k_minus_bb"] == 90


def test_parse_personal_profile():
    html = """
    <html><body>
      <table>
        <tr><td>ポジション</td><td>投手</td></tr>
        <tr><td>投打</td><td>右投右打</td></tr>
        <tr><td>身長／体重</td><td>183cm／86kg</td></tr>
        <tr><td>生年月日</td><td>1998年3月9日</td></tr>
        <tr><td>経歴</td><td>高校 - 大学 - 球団</td></tr>
      </table>
    </body></html>
    """
    profile = ctx._parse_personal_profile(html)
    assert profile["position"] == "投手"
    assert profile["bats_throws"] == "右投右打"
    assert profile["height_weight"] == "183cm／86kg"


def test_parse_year_rows():
    html = """
    <table>
      <tr><th>年度</th><th>所属球団</th><th>登板</th><th>勝利</th><th>投球回</th><th>三振</th><th>防御率</th></tr>
      <tr><td>2026</td><td>チーム</td><td>16</td><td>10</td><td>104</td><td>89</td><td>2.25</td></tr>
    </table>
    """
    tables = ctx._parse_year_rows(html)
    assert len(tables["pitching"]) == 1
    assert tables["pitching"][0]["年度"] == "2026"
    assert tables["pitching"][0]["投球回_num"] == 104.0


def test_resolve_player_refs_exact():
    ref = ctx.PlayerRef(
        player_id="123",
        name="テスト太郎",
        url="https://npb.jp/bis/players/123.html",
        team=None,
        position="投手",
        roster_source="https://npb.jp/bis/players/active/index_te.html",
        roster_available_at_utc="2026-10-04T00:00:00+00:00",
    )
    got = ctx.resolve_player_refs(["テスト太郎"], {"テスト太郎": [ref]})
    assert got["テスト太郎"].player_id == "123"
