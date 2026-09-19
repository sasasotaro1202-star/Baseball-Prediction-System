from production_npb import build_target_rows

def test_20260920_has_six_pit_safe_games():
    d=build_target_rows("2026-09-20")
    assert len(d)==6
    assert d["confirmed_starters"].all()
    assert (d["starter_evidence_status"]=="official_announced").all()
    assert d["home_score"].isna().all()
    assert d["away_score"].isna().all()
