from pathlib import Path
from production_npb import (
    _official_daily_competition_metadata,
    build_target_rows,
    parse_official_starters_html,
)

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
    monkeypatch.setattr(p, "_official_daily_start_times", lambda target_date, **kwargs: {
        ("読売ジャイアンツ", "東京ヤクルトスワローズ"): "14:00",
        ("中日ドラゴンズ", "広島東洋カープ"): "18:00",
        ("阪神タイガース", "横浜DeNAベイスターズ"): "18:00",
        ("北海道日本ハムファイターズ", "オリックス・バファローズ"): "14:00",
        ("東北楽天ゴールデンイーグルス", "福岡ソフトバンクホークス"): "14:00",
        ("千葉ロッテマリーンズ", "埼玉西武ライオンズ"): "18:00",
    })
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
        "_official_daily_start_times",
        lambda target_date, **kwargs: {("読売ジャイアンツ", "阪神タイガース"): "18:00"},
    )
    monkeypatch.setattr(
        p,
        "official_starters",
        lambda target_date, **kwargs: [{
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


def test_official_starter_snapshot_rejects_future_retrieval(monkeypatch, tmp_path):
    import json
    import production_npb as p

    root = tmp_path
    snapshot_dir = root / "data" / "official_starters"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "2026-09-20.json").write_text(
        json.dumps({
            "schema_version": "npb-official-starter-snapshot-v1",
            "target_date": "2026-09-20",
            "source_type": "NPB_OFFICIAL",
            "source_url": "https://npb.jp/",
            "retrieved_at_utc": "2999-01-01T00:00:00+00:00",
            "games": [{
                "home": "読売ジャイアンツ",
                "away": "阪神タイガース",
                "home_starter": "投手A",
                "away_starter": "投手B",
                "official_start_time": "18:00",
            }],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(p, "ROOT", root)
    try:
        p._load_official_starter_snapshot("2026-09-20")
    except RuntimeError as exc:
        assert "in the future" in str(exc)
    else:
        raise AssertionError("future-dated official snapshot was accepted")


def test_existing_malformed_official_snapshot_is_not_silently_ignored(monkeypatch, tmp_path):
    import json
    import production_npb as p

    root = tmp_path
    snapshot_dir = root / "data" / "official_starters"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "2026-09-20.json").write_text(
        json.dumps({
            "schema_version": "BROKEN",
            "target_date": "2026-09-20",
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(p, "ROOT", root)

    try:
        p.official_starters("2026-09-20")
    except RuntimeError as exc:
        assert "schema mismatch" in str(exc)
    else:
        raise AssertionError("malformed existing official snapshot was silently ignored")


def test_daily_schedule_time_parser_accepts_pair_time_stream(monkeypatch):
    import production_npb as p

    html = """
    <html><body>
      <script>var hiddenClock = "3:05";</script>
      <img alt="広島東洋カープ">
      <span>（マツダスタジアム）</span><span>18:00</span>
      <img alt="読売ジャイアンツ">
      <img alt="北海道日本ハムファイターズ">
      <span>（エスコンＦ）</span><span>18:00</span>
      <img alt="東北楽天ゴールデンイーグルス">
    </body></html>
    """
    monkeypatch.setattr(p, "fetch_text", lambda url: html)
    got = p._official_daily_start_times("2026-09-24")
    assert got[("広島東洋カープ", "読売ジャイアンツ")] == "18:00"
    assert got[("北海道日本ハムファイターズ", "東北楽天ゴールデンイーグルス")] == "18:00"


def test_target_rows_preserves_prediction_time_competition_metadata(monkeypatch):
    import production_npb as p
    import pandas as pd

    monkeypatch.setattr(
        p,
        "official_starters",
        lambda target_date, **kwargs: [{
            "home": "広島東洋カープ",
            "away": "読売ジャイアンツ",
            "home_starter": "投手A",
            "away_starter": "投手B",
            "confirmed_starters": True,
            "starter_evidence_status": "official_announced",
            "starter_source": "https://npb.jp/announcement/starter/",
            "official_start_time": "18:00",
        }],
    )
    monkeypatch.setattr(
        p,
        "_official_daily_start_times",
        lambda target_date, **kwargs: (
            kwargs["metadata_out"].update({
                "competition": "npb_regular",
                "stage": "regular_season",
                "season_type": "regular_season",
                "game_class": "official",
                "competition_key": "NPB:npb_regular:regular_season",
                "status": "classified",
                "source_url": "https://npb.jp/bis/eng/2026/games/gm20260924.html",
                "source_field": "npb_daily_schedule_heading",
                "source_value": "公式戦【試合予定】",
            }) or {("広島東洋カープ", "読売ジャイアンツ"): "18:00"}
        ),
    )
    monkeypatch.setattr(
        p,
        "_utc_now",
        lambda: pd.Timestamp("2026-09-23 12:00:00+00:00"),
    )

    rows = p.build_target_rows("2026-09-24")
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["competition"] == "npb_regular"
    assert row["competition_stage"] == "regular_season"
    assert row["season_type"] == "regular_season"
    assert row["game_class"] == "official"
    assert row["competition_key"] == "NPB:npb_regular:regular_season"
    assert row["competition_classification_status"] == "classified"
    assert row["competition_metadata_source"] == "https://npb.jp/bis/eng/2026/games/gm20260924.html"
    assert row["competition_metadata_source_field"] == "npb_daily_schedule_heading"
    assert row["competition_metadata_source_value"] == "公式戦【試合予定】"


def test_target_rows_uses_official_daily_schedule_time_as_authoritative(monkeypatch):
    import production_npb as p
    import pandas as pd

    monkeypatch.setattr(
        p,
        "official_starters",
        lambda target_date, **kwargs: [{
            "home": "広島東洋カープ",
            "away": "読売ジャイアンツ",
            "home_starter": "投手A",
            "away_starter": "投手B",
            "confirmed_starters": True,
            "starter_evidence_status": "official_announced",
            "starter_source": "https://npb.jp/announcement/starter/",
            "official_start_time": "03:05",
        }],
    )
    monkeypatch.setattr(
        p,
        "_official_daily_start_times",
        lambda target_date, **kwargs: {("広島東洋カープ", "読売ジャイアンツ"): "18:00"},
    )
    monkeypatch.setattr(
        p,
        "_utc_now",
        lambda: pd.Timestamp("2026-09-23 12:00:00+00:00"),
    )
    rows = p.build_target_rows("2026-09-24")
    assert len(rows) == 1
    assert rows.iloc[0]["official_start_time"] == "18:00"
    assert str(rows.iloc[0]["start_time_source"]).startswith("https://npb.jp/bis/eng/2026/games/")




def test_starter_time_parser_prefers_structural_game_card_over_average_duration():
    import production_npb as p

    html = """
    <div class="unit">
      <img alt="広島東洋カープ"><div class="team_left"><span>森下 暢仁</span></div>
      <img alt="読売ジャイアンツ"><div class="team_left"><span>西舘 勇陽</span></div>
      <span>（マツダスタジアム）18:00</span>
    </div>
    <div>2026年 平均試合時間（9/22） 3:05 （9回試合のみ）</div>
    """
    rows = p.parse_official_starters_html(
        '<h4>9月24日の予告先発投手</h4>' + html,
        "2026-09-24",
    )
    assert len(rows) == 1
    assert rows[0]["official_start_time"] == "18:00"



def test_target_rows_uses_actual_information_cutoff_and_keeps_30m_as_preferred(monkeypatch):
    import production_npb as p
    import pandas as pd

    monkeypatch.setattr(
        p,
        "official_starters",
        lambda target_date, **kwargs: [{
            "home": "広島東洋カープ",
            "away": "読売ジャイアンツ",
            "home_starter": "投手A",
            "away_starter": "投手B",
            "confirmed_starters": True,
            "starter_evidence_status": "official_announced",
            "starter_source": "https://npb.jp/announcement/starter/",
            "official_start_time": "18:00",
        }],
    )
    monkeypatch.setattr(
        p,
        "_official_daily_start_times",
        lambda target_date, **kwargs: {("広島東洋カープ", "読売ジャイアンツ"): "18:00"},
    )
    monkeypatch.setattr(
        p,
        "_utc_now",
        lambda: pd.Timestamp("2026-09-20 08:30:01+00:00"),
    )
    rows = p.build_target_rows(
        "2026-09-20",
        minimum_lead_minutes=0.0,
        maximum_lead_minutes=60.0,
        preferred_lead_minutes=30.0,
    )
    assert len(rows) == 1
    assert rows.iloc[0]["prediction_cutoff_utc"] == "2026-09-20T08:30:01+00:00"
    assert rows.iloc[0]["prediction_deadline_utc"] == "2026-09-20T08:30:00+00:00"
    assert rows.iloc[0]["preferred_prediction_cutoff_utc"] == "2026-09-20T08:30:00+00:00"
    assert bool(rows.iloc[0]["preferred_30m_met"]) is False
    assert float(rows.iloc[0]["lead_minutes_at_generation"]) < 30.0
    assert rows.iloc[0]["starter_evidence_observed_at_utc"] == "2026-09-20T08:30:01+00:00"
    assert rows.iloc[0]["starter_source"] == "https://npb.jp/announcement/starter/"


def test_pregame_only_limits_prediction_window_to_upcoming_60_minutes(monkeypatch):
    import production_npb as p
    import pandas as pd

    monkeypatch.setattr(
        p,
        "official_starters",
        lambda target_date, **kwargs: [{
            "home": "広島東洋カープ",
            "away": "読売ジャイアンツ",
            "home_starter": "投手A",
            "away_starter": "投手B",
            "confirmed_starters": True,
            "starter_evidence_status": "official_announced",
            "starter_source": "https://npb.jp/announcement/starter/",
            "official_start_time": "18:00",
        }],
    )
    monkeypatch.setattr(
        p,
        "_official_daily_start_times",
        lambda target_date, **kwargs: {("広島東洋カープ", "読売ジャイアンツ"): "18:00"},
    )
    monkeypatch.setattr(
        p,
        "_utc_now",
        lambda: pd.Timestamp("2026-09-20 07:45:00+00:00"),
    )
    # 75 minutes before first pitch: pregame-only should defer it to a later
    # 5-minute scheduler tick rather than predict too early.
    assert p.build_target_rows("2026-09-20", minimum_lead_minutes=0, maximum_lead_minutes=60).empty

    # 29 minutes before first pitch: 30m is preferred, but the pregame path
    # still permits a valid PIT-safe forecast rather than missing the game.
    monkeypatch.setattr(
        p,
        "_utc_now",
        lambda: pd.Timestamp("2026-09-20 08:31:00+00:00"),
    )
    rows = p.build_target_rows(
        "2026-09-20",
        minimum_lead_minutes=0,
        maximum_lead_minutes=60,
        preferred_lead_minutes=30,
    )
    assert len(rows) == 1
    assert bool(rows.iloc[0]["preferred_30m_met"]) is False


def test_direct_npb_production_entrypoint_enforces_registry_gate(tmp_path, monkeypatch):
    import production_npb as p

    called = []
    monkeypatch.setattr(p, "build_target_rows", lambda *args, **kwargs: called.append(True))
    result = p.predict("2026-10-02", str(tmp_path))
    assert result["execution_status"] == "BLOCKED_PRODUCTION_GATE"
    assert result["predictions"] == []
    assert called == []



def test_daily_schedule_competition_metadata_uses_semantic_heading_not_navigation_links():
    html = """
    <html><body>
      <nav>
        <a>日本シリーズ</a>
        <a>クライマックスシリーズ</a>
        <a>オールスター・ゲーム</a>
        <a>ファーム日本選手権</a>
      </nav>
      <h3>公式戦【試合予定】</h3>
    </body></html>
    """
    metadata = _official_daily_competition_metadata(
        html,
        "https://npb.jp/bis/eng/2026/games/gm20261003.html",
    )
    assert metadata["competition"] == "npb_regular"
    assert metadata["stage"] == "regular_season"
    assert metadata["season_type"] == "regular_season"
    assert metadata["game_class"] == "official"
    assert metadata["competition_key"] == "NPB:npb_regular:regular_season"
    assert metadata["status"] == "classified"
    assert metadata["source_field"] == "npb_daily_schedule_heading"
    assert metadata["source_value"] == "公式戦【試合予定】"


def test_daily_schedule_competition_metadata_classifies_interleague_without_outcome_data():
    html = "<h3>交流戦【試合予定】</h3>"
    metadata = _official_daily_competition_metadata(
        html,
        "https://npb.jp/bis/eng/2026/games/gm20260617.html",
    )
    assert metadata["competition"] == "npb_interleague"
    assert metadata["stage"] == "interleague"
    assert metadata["season_type"] == "regular_season"
    assert metadata["game_class"] == "official"
    assert metadata["status"] == "classified"


def test_daily_schedule_competition_metadata_fails_closed_when_heading_unknown():
    html = """
    <nav><a>日本シリーズ</a><a>交流戦</a></nav>
    <h3>イベント日程【試合予定】</h3>
    """
    metadata = _official_daily_competition_metadata(
        html,
        "https://npb.jp/bis/eng/2026/games/gm20261003.html",
    )
    assert metadata["competition"] == "npb_unknown"
    assert metadata["stage"] == "unknown"
    assert metadata["season_type"] == "unknown"
    assert metadata["game_class"] == "unknown"
    assert metadata["competition_key"] == "NPB:npb_unknown:unknown"
    assert metadata["status"] == "unknown"


def test_daily_schedule_metadata_path_is_outcome_free():
    html = """
    <h3>日本シリーズ【試合予定】</h3>
    <div>0 - 99</div>
    """
    metadata = _official_daily_competition_metadata(
        html,
        "https://npb.jp/bis/eng/2026/games/gm1010.html",
    )
    assert metadata["competition"] == "npb_japan_series"
    assert metadata["status"] == "classified"



def test_daily_schedule_competition_metadata_ignores_non_schedule_headings():
    html = """
    <h3>公式戦成績</h3>
    <h3>日本シリーズ【試合予定】</h3>
    """
    metadata = _official_daily_competition_metadata(
        html,
        "https://npb.jp/bis/2026/games/gm20261101.html",
    )
    assert metadata["competition"] == "npb_japan_series"
    assert metadata["stage"] == "japan_series"
