"""Cross-age and cross-competition baseball scope catalog.

This module defines the research surface independently of any one league model.
Rules are intentionally coarse where competition-specific confirmation is still
needed. Unknown rule details remain explicit instead of being inferred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CompetitionScope:
    scope_id: str
    label: str
    level: str
    gender: str
    outcome_contract: str
    rule_family: str
    source_ids: tuple[str, ...]
    priority: int = 2


SCOPES: tuple[CompetitionScope, ...] = (
    CompetitionScope("NPB", "NPB", "professional", "mixed", "three_way", "npb_standard",
                     ("npb_schedule", "npb_starters", "npb_lineups", "npb_hawkeye_npbplus",
                      "npb_public_spaia_pbp", "armstjc_npb_repository", "weather"), 1),
    CompetitionScope("MLB", "MLB", "professional", "mixed", "binary", "mlb_standard",
                     ("mlb_statsapi", "mlb_starters", "statcast", "weather"), 1),
    CompetitionScope("MiLB", "MLB Minor League", "professional", "mixed", "competition_defined",
                     "minor_league_specific", ("mlb_milb_statcast", "mlb_statsapi"), 1),
    CompetitionScope("KBO", "KBO", "professional", "mixed", "competition_defined", "kbo_standard",
                     ("kbo_official_tracking", "kbo_official_stats", "kbo_naver_pbp_public"), 1),
    CompetitionScope("CPBL", "CPBL", "professional", "mixed", "competition_defined", "cpbl_standard",
                     ("cpbl_rebas",), 1),
    CompetitionScope("LMB", "Liga Mexicana de Béisbol", "professional", "mixed", "competition_defined",
                     "lmb_standard", ("lmb_official",), 2),
    CompetitionScope("ABL", "Australian Baseball League", "professional", "mixed", "competition_defined",
                     "abl_standard", ("abl_official",), 2),
    CompetitionScope("NCAA_D1", "NCAA Division I", "college", "mixed", "competition_defined",
                     "ncaa_standard", ("ncaa_baseball_sportsdataverse",), 1),
    CompetitionScope("Japan_University", "Japan University Baseball", "college", "mixed",
                     "competition_defined", "japan_amateur_specific",
                     ("omyu_university", "omyu_jaba"), 1),
    CompetitionScope("Japan_HighSchool", "Japan High School", "high_school", "mixed",
                     "competition_defined", "japan_high_school_specific",
                     ("omyu_high_school", "jhbf_official"), 1),
    CompetitionScope("Japan_WomensHighSchool", "Japan Women's High School", "high_school", "female",
                     "competition_defined", "japan_high_school_specific",
                     ("omyu_womens_high_school",), 1),
    CompetitionScope("Japan_Junior_LittleSenior", "Japan Little Senior", "junior", "mixed",
                     "competition_defined", "japan_junior_specific",
                     ("omyu_little_senior", "omyu_junior"), 1),
    CompetitionScope("Japan_Junior_Boys", "Japan Boys League", "junior", "mixed",
                     "competition_defined", "japan_junior_specific",
                     ("omyu_boys", "omyu_junior"), 1),
    CompetitionScope("Japan_Junior_Young", "Japan Young League", "junior", "mixed",
                     "competition_defined", "japan_junior_specific",
                     ("omyu_young", "omyu_junior"), 1),
    CompetitionScope("Japan_Junior_Pony", "Japan Pony League", "junior", "mixed",
                     "competition_defined", "japan_junior_specific",
                     ("omyu_pony", "omyu_junior"), 1),
    CompetitionScope("Japan_Elementary", "Japan Elementary / 学童", "elementary", "mixed",
                     "competition_defined", "japan_elementary_specific",
                     ("omyu_elementary",), 2),
    CompetitionScope("Japan_WomensJunior", "Japan Women's Junior", "junior", "female",
                     "competition_defined", "japan_junior_specific",
                     ("omyu_womens_junior", "omyu_youth_girls"), 1),
    CompetitionScope("Japan_U12", "Japan U-12 National Team", "international_age_group", "mixed",
                     "competition_defined", "wbsc_age_group_specific",
                     ("samurai_youth", "wbsc_age_group_reports"), 1),
    CompetitionScope("Japan_U15", "Japan U-15 National Team", "international_age_group", "mixed",
                     "competition_defined", "wbsc_age_group_specific",
                     ("samurai_youth", "wbsc_age_group_reports"), 1),
    CompetitionScope("Japan_U18", "Japan U-18 National Team", "international_age_group", "mixed",
                     "competition_defined", "wbsc_age_group_specific",
                     ("samurai_youth", "wbsc_age_group_reports"), 1),
    CompetitionScope("Japan_U23", "Japan U-23 / university-age international", "international_age_group",
                     "mixed", "competition_defined", "wbsc_age_group_specific",
                     ("samurai_u23", "wbsc_age_group_reports"), 2),
    CompetitionScope("WBSC_U12", "WBSC U-12", "international_age_group", "mixed",
                     "competition_defined", "wbsc_age_group_specific",
                     ("wbsc_mywbsc", "wbsc_age_group_reports"), 1),
    CompetitionScope("WBSC_U15", "WBSC U-15", "international_age_group", "mixed",
                     "competition_defined", "wbsc_age_group_specific",
                     ("wbsc_mywbsc", "wbsc_age_group_reports"), 1),
    CompetitionScope("WBSC_U18", "WBSC U-18", "international_age_group", "mixed",
                     "competition_defined", "wbsc_age_group_specific",
                     ("wbsc_mywbsc", "wbsc_age_group_reports"), 1),
    CompetitionScope("WBSC_U23", "WBSC U-23", "international_age_group", "mixed",
                     "competition_defined", "wbsc_age_group_specific",
                     ("wbsc_mywbsc", "wbsc_age_group_reports"), 1),
)


def scopes() -> tuple[CompetitionScope, ...]:
    return SCOPES


def get_scope(scope_id: str) -> CompetitionScope:
    for scope in SCOPES:
        if scope.scope_id == scope_id:
            return scope
    raise KeyError(f"unknown competition scope: {scope_id}")


def catalog_rows() -> list[dict[str, Any]]:
    return [asdict(s) for s in SCOPES]
