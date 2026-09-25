from research.competition_taxonomy import classify_mlb, classify_npb, classify_game


def test_npb_interleague_is_regular_season_but_separate_partition():
    label = classify_npb("公式戦・交流戦")
    assert label.status == "classified"
    assert label.competition == "npb_interleague"
    assert label.season_type == "regular_season"
    assert label.stage == "interleague"
    assert label.competition_key == "NPB:npb_interleague:interleague"


def test_npb_postseason_is_separate():
    assert classify_npb("クライマックスシリーズ").competition == "npb_climax_series"
    assert classify_npb("日本シリーズ").stage == "japan_series"


def test_npb_exhibition_is_not_regular():
    label = classify_npb("オープン戦")
    assert label.season_type == "spring"
    assert label.game_class == "exhibition"


def test_mlb_game_type_partitioning():
    assert classify_mlb("R").stage == "regular_season"
    assert classify_mlb("S").season_type == "spring"
    assert classify_mlb("F").stage == "wild_card"
    assert classify_mlb("D").stage == "division_series"
    assert classify_mlb("L").stage == "league_championship_series"
    assert classify_mlb("W").game_class == "championship"
    assert classify_mlb("A").season_type == "all_star"


def test_unknown_is_fail_closed():
    label = classify_mlb("", "World Series Game 1")
    assert label.status == "unknown"
    assert label.competition == "mlb_unknown"


def test_dispatch():
    assert classify_game("NPB", game_type="交流戦").league == "NPB"
    assert classify_game("MLB", game_type="R").league == "MLB"
    assert classify_game("foo").status == "unknown"
