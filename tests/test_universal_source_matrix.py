import pandas as pd
import pytest

from data.source_registry import SOURCES
from research.universal_source_matrix import SOURCE_CAPABILITIES

from research.competition_catalog import get_scope, scopes
from research.universal_source_matrix import application_matrix, application_plan, pit_safe_rows


def test_catalog_covers_multiple_age_levels():
    levels = {s.level for s in scopes()}
    assert {"professional", "college", "high_school", "junior", "elementary", "international_age_group"} <= levels


def test_high_school_plan_includes_public_research_sources():
    rows = application_plan("Japan_HighSchool")
    source_ids = {r["source_id"] for r in rows}
    assert "omyu_high_school" in source_ids
    assert "jhbf_official" in source_ids
    assert all(r["pit_requirement"] == "explicit available_at <= prediction_time" for r in rows)


def test_matrix_contains_every_catalog_scope():
    rows = application_matrix()
    scopes_seen = {r["scope_id"] for r in rows}
    assert scopes_seen == {s.scope_id for s in scopes()}


def test_pit_filter_fails_closed_without_timestamps():
    with pytest.raises(ValueError):
        pit_safe_rows(pd.DataFrame([{"value": 1}]))


def test_pit_filter_rejects_future_and_unknown():
    frame = pd.DataFrame([
        {"id": "pass", "available_at": "2026-09-01T00:00:00Z", "prediction_time": "2026-09-02T00:00:00Z", "status": "KNOWN"},
        {"id": "future", "available_at": "2026-09-03T00:00:00Z", "prediction_time": "2026-09-02T00:00:00Z", "status": "KNOWN"},
        {"id": "unknown", "available_at": "2026-09-01T00:00:00Z", "prediction_time": "2026-09-02T00:00:00Z", "status": "UNVERIFIED"},
        {"id": "missing", "available_at": None, "prediction_time": "2026-09-02T00:00:00Z", "status": "KNOWN"},
    ])
    out = pit_safe_rows(frame)
    assert out["id"].tolist() == ["pass"]


def test_scope_rule_metadata_is_explicit():
    scope = get_scope("Japan_U12")
    assert scope.outcome_contract == "competition_defined"
    assert scope.rule_family == "wbsc_age_group_specific"


def test_every_registered_source_has_explicit_capability_mapping():
    registered = {s.source_id for s in SOURCES}
    assert not (registered - set(SOURCE_CAPABILITIES)), (
        f"unmapped registered sources: {sorted(registered - set(SOURCE_CAPABILITIES))}"
    )


def test_mlb_only_tracking_sources_are_not_applied_to_high_school():
    source_ids = {r["source_id"] for r in application_plan("Japan_HighSchool")}
    assert "statcast" not in source_ids
    assert "fangraphs" not in source_ids
    assert "retrosheet" not in source_ids


def test_every_registered_source_is_applied_to_at_least_one_scope():
    registered = {s.source_id for s in SOURCES}
    applied = {r["source_id"] for r in application_matrix()}
    assert registered <= applied, (
        f"registered sources not applied to any scope: {sorted(registered - applied)}"
    )


def test_global_sources_have_expected_scope_bindings():
    checks = {
        "WBC": "wbc_official_stats",
        "WBSC_WomensBaseball": "wbsc_womens_baseball",
        "LittleLeague_WorldSeries": "little_league_world_series",
        "CapeCod": "cape_cod_league",
        "WBSC_Europe": "wbsc_europe_baseball",
        "LIDOM": "lidom_mlb_winter",
        "LVBP": "lvbp_official",
        "LBPRC": "lbprc_official",
        "LMP": "lmp_mlb_winter",
    }
    for scope_id, source_id in checks.items():
        assert source_id in {r["source_id"] for r in application_plan(scope_id)}


def test_domestic_amateur_scopes_bind_official_sources():
    assert "jaba_official" in {r["source_id"] for r in application_plan("Japan_Amateur_JABA")}
    assert "big6_scorebook" in {r["source_id"] for r in application_plan("Japan_University")}


def test_netherlands_youth_scope_is_explicit():
    rows = application_plan("Netherlands_Youth")
    assert "knbsb_baseball" in {r["source_id"] for r in rows}


def test_czech_baseball_scope_binds_official_source():
    assert "czech_baseball_assoc" in {r["source_id"] for r in application_plan("Czechia_Baseball")}


def test_milb_levels_bind_milb_repository():
    for scope_id in ("MiLB_AAA", "MiLB_AA", "MiLB_APlus", "MiLB_A", "MiLB_Rookie"):
        assert "milb_data_repository" in {r["source_id"] for r in application_plan(scope_id)}


def test_new_international_sources_bind():
    assert "ffbs_d1_official" in {r["source_id"] for r in application_plan("France_D1")}
    assert "wbc_scouting_public_dataset" in {r["source_id"] for r in application_plan("WBC_PlayerPrior")}


def test_germany_and_spain_age_scopes_bind_sources():
    for scope_id, source_id in (
        ("Germany_DBL", "dbv_dbl_official"),
        ("Germany_2BL", "dbsv_2bundesliga"),
        ("Germany_U18", "bbsv_youth_baseball"),
        ("Spain_Division_Honor", "rfebs_baseball"),
        ("Spain_U18", "rfebs_baseball"),
        ("Spain_U15", "rfebs_baseball"),
        ("Spain_U12", "rfebs_baseball"),
        ("Colombia_LPBC", "lpbc_colombia"),
    ):
        assert source_id in {r["source_id"] for r in application_plan(scope_id)}


def test_us_summer_collegiate_scopes_bind_sources():
    assert "northwoods_league" in {r["source_id"] for r in application_plan("Northwoods")}
    assert "west_coast_league" in {r["source_id"] for r in application_plan("WestCoastLeague")}


def test_every_scope_reference_is_registered():
    registered = {s.source_id for s in SOURCES}
    referenced = {source_id for scope in scopes() for source_id in scope.source_ids}
    assert referenced <= registered, (
        f"scope references not in registry: {sorted(referenced - registered)}"
    )


def test_canada_and_great_britain_multi_age_scopes_bind_sources():
    checks = {
        "Canada_ICBA": "icba_ontario",
        "Canada_BaseballQuebec": "baseball_quebec",
        "Canada_WomensBaseball": "lfbq_womens",
        "GreatBritain_Senior": "bbf_senior_leagues",
        "GreatBritain_Youth": "bbf_youth_u16_u18",
        "Philippines_National": "paba_philippines",
        "AsianGames_2026": "asian_games_baseball",
    }
    for scope_id, source_id in checks.items():
        assert source_id in {r["source_id"] for r in application_plan(scope_id)}


def test_australia_and_puerto_rico_youth_scopes_bind_sources():
    checks = {
        "Australia_U18": "baseball_australia_youth",
        "Australia_U16": "baseball_australia_youth",
        "Australia_Womens": "baseball_australia_womens",
        "Australia_Womens_U16": "baseball_australia_womens_u16",
        "PuertoRico_Juvenile_DAA": "lbda_juvenil_puerto_rico",
    }
    for scope_id, source_id in checks.items():
        assert source_id in {r["source_id"] for r in application_plan(scope_id)}


def test_brazil_multi_age_scopes_bind_cbbs():
    for scope_id in ("Brazil_PreInfantil", "Brazil_Infantil", "Brazil_Juvenil", "Brazil_Junior"):
        assert "cbbs_brazil_competitions" in {r["source_id"] for r in application_plan(scope_id)}


def test_israel_all_age_scopes_bind_iab():
    for scope_id in ("Israel_Minors", "Israel_Juveniles", "Israel_LittleLeague", "Israel_Cadets", "Israel_Juniors", "Israel_Premier"):
        assert "iab_israel_baseball" in {r["source_id"] for r in application_plan(scope_id)}


def test_argentina_lab_binds_official_source():
    assert "liga_argentina_beisbol" in {r["source_id"] for r in application_plan("Argentina_LAB")}


def test_ncaa_divisions_are_explicitly_scoped():
    assert "ncaa_baseball_d1" in {r["source_id"] for r in application_plan("NCAA_D1")}
    assert "ncaa_baseball_d2" in {r["source_id"] for r in application_plan("NCAA_D2")}
    assert "ncaa_baseball_d3" in {r["source_id"] for r in application_plan("NCAA_D3")}

def test_kbo_futures_scope_is_explicit():
    assert "kbo_official_stats" in {r["source_id"] for r in application_plan("KBO_Futures")}

def test_espn_sources_are_bound_to_supported_scopes():
    assert "espn_mlb" in {r["source_id"] for r in application_plan("MLB")}
    assert "espn_college_baseball" in {r["source_id"] for r in application_plan("NCAA_D1")}
    assert "espn_college_baseball" in {r["source_id"] for r in application_plan("NCAA_D2")}
    assert "espn_college_baseball" in {r["source_id"] for r in application_plan("NCAA_D3")}
    assert "espn_international" in {r["source_id"] for r in application_plan("WBC")}
