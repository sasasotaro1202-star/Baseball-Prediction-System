"""Canonical competition registry for baseball research and production gating.

The registry is metadata-only: listing a competition never makes it production
eligible. Each competition family gets an explicit outcome contract and rules
profile so league play and tournament/knockout play are never mixed blindly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal["PRODUCTION_ELIGIBLE", "RESEARCH_ONLY", "UNAVAILABLE"]


@dataclass(frozen=True)
class CompetitionSpec:
    competition_id: str
    name: str
    competition_type: str
    level: str
    gender: str
    geography: str
    outcome_contract: str
    status: Status
    requires_starter_announcement: bool
    pit_required: bool = True
    phase_type: str = "league"
    rules_profile: str = "standard_baseball"
    notes: str = ""
    discovery_priority: int = 100
    discovery_url: str = ""
    starter_source_id: str = ""


COMPETITIONS: tuple[CompetitionSpec, ...] = (
    # League / regular-season competitions: keep these isolated and highest priority.
    # NPB remains fail-closed until the real-data candidate lifecycle has produced
    # an independent holdout-backed ADOPT decision. This prevents a metadata-only
    # registry entry from becoming a production promotion by accident.
    CompetitionSpec("NPB", "Nippon Professional Baseball", "professional", "top", "mixed", "Japan", "HOME_DRAW_AWAY", "RESEARCH_ONLY", True, phase_type="league", rules_profile="npb", notes="Production promotion requires candidate lock, independent holdout, PIT starter evidence, calibration, score/Low-High checks, and explicit ADOPT evidence."),
    CompetitionSpec("MLB", "Major League Baseball", "professional", "top", "mixed", "United States/Canada", "HOME_AWAY", "RESEARCH_ONLY", True, phase_type="league", rules_profile="mlb", notes="Historical starter announcement timestamp evidence remains a production gate."),

    # High-volume professional competitions being brought into the scope frontier.
    # These remain RESEARCH_ONLY until isolated PIT/OOS/holdout gates pass.
    CompetitionSpec(
        "KBO", "Korea Baseball Organization", "professional", "top", "mixed", "South Korea",
        "HOME_DRAW_AWAY", "RESEARCH_ONLY", True,
        phase_type="league", rules_profile="kbo", discovery_priority=1,
        discovery_url="https://eng.koreabaseball.com/Schedule/DailySchedule.aspx",
        starter_source_id="kbo_official_stats",
        notes="Official 2026 schedule is public; historical starter-announcement timing and feature PIT require dedicated validation.",
    ),
    CompetitionSpec(
        "CPBL", "Chinese Professional Baseball League", "professional", "top", "mixed", "Taiwan",
        "HOME_DRAW_AWAY", "RESEARCH_ONLY", True,
        phase_type="league", rules_profile="cpbl", discovery_priority=2,
        discovery_url="https://stats.cpbl.com.tw/schedule/2026-A-",
        starter_source_id="cpbl_rebas",
        notes="Official 2026 schedule/standings are public; starter and historical PIT evidence require validation.",
    ),
    CompetitionSpec(
        "LMB", "Liga Mexicana de Beisbol", "professional", "top", "mixed", "Mexico",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", True,
        phase_type="league", rules_profile="lmb", discovery_priority=3,
        discovery_url="https://lmb.com.mx/noticias/calendario-oficial-de-la-temporada-2026-de-la-liga-mexicana-de-beisbol",
        starter_source_id="lmb_official",
        notes="Official 2026 schedule is public; structured pregame/starter PIT requires validation.",
    ),
    CompetitionSpec(
        "ABL", "Australian Baseball League", "professional", "top", "mixed", "Australia",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", True,
        phase_type="league", rules_profile="abl", discovery_priority=4,
        discovery_url="https://plus.baseball.com.au/en-int/page/home",
        starter_source_id="abl_official",
        notes="Official ABL site exposes upcoming games; historical PIT and structured starter data require validation.",
    ),
    CompetitionSpec(
        "MILB", "Minor League Baseball", "professional_development", "development", "mixed", "United States/Canada",
        "HOME_AWAY", "RESEARCH_ONLY", False,
        phase_type="league", rules_profile="milb", discovery_priority=5,
        discovery_url="https://www.mlb.com/milb",
        starter_source_id="milb_data_repository",
        notes="Official MiLB schedules/statistics are public; competition/level-specific PIT and starter conventions require validation.",
    ),
    CompetitionSpec(
        "KBO_FUTURES", "KBO Futures League", "professional_development", "development", "mixed", "South Korea",
        "HOME_DRAW_AWAY", "RESEARCH_ONLY", True,
        phase_type="league", rules_profile="kbo_futures", discovery_priority=6,
        discovery_url="https://www.koreabaseball.com/Schedule/Futures/Index.aspx",
        starter_source_id="kbo_official_stats",
        notes="Separate from KBO top league; official 2026 schedule/results are public and must not be mixed with KBO.",
    ),
    CompetitionSpec(
        "JAPAN_INDEPENDENT", "Japanese Independent Leagues", "professional_independent", "senior", "mixed", "Japan",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", True,
        phase_type="league", rules_profile="japan_independent", discovery_priority=7,
        discovery_url="https://data.iblj.co.jp/",
        starter_source_id="iblj_official_stats",
        notes="Independent leagues are tracked separately; structured PIT/starter coverage must be validated per league.",
    ),
    CompetitionSpec(
        "LIDOM", "Liga de Béisbol Profesional de la República Dominicana", "winter_league", "senior", "mixed", "Dominican Republic",
        "HOME_AWAY", "RESEARCH_ONLY", True,
        phase_type="league_and_tournament", rules_profile="winter_caribbean", discovery_priority=8,
        discovery_url="https://www.lidom.com/",
        starter_source_id="lidom_mlb_winter",
    ),
    CompetitionSpec(
        "LMP", "Liga Mexicana del Pacífico", "winter_league", "senior", "mixed", "Mexico",
        "HOME_AWAY", "RESEARCH_ONLY", True,
        phase_type="league_and_tournament", rules_profile="winter_mexico", discovery_priority=9,
        discovery_url="https://www.lmp.mx/",
        starter_source_id="lmp_mlb_winter",
    ),
    CompetitionSpec(
        "LVBP", "Liga Venezolana de Béisbol Profesional", "winter_league", "senior", "mixed", "Venezuela",
        "HOME_AWAY", "RESEARCH_ONLY", True,
        phase_type="league_and_tournament", rules_profile="winter_caribbean", discovery_priority=10,
        discovery_url="https://stats.lvbp.com/",
        starter_source_id="lvbp_official",
    ),
    CompetitionSpec(
        "LBPRC", "Liga de Béisbol Profesional Roberto Clemente", "winter_league", "senior", "mixed", "Puerto Rico",
        "HOME_AWAY", "RESEARCH_ONLY", True,
        phase_type="league_and_tournament", rules_profile="winter_caribbean", discovery_priority=11,
        discovery_url="https://www.ligapr.com/",
        starter_source_id="lbprc_official",
    ),
    CompetitionSpec(
        "CCBL", "Cape Cod Baseball League", "summer_collegiate", "college", "mixed", "United States",
        "HOME_AWAY", "RESEARCH_ONLY", True,
        phase_type="league", rules_profile="college_summer", discovery_priority=20,
        discovery_url="https://www.capecodbaseball.org/",
        starter_source_id="cape_cod_league",
    ),
    CompetitionSpec(
        "WBSC_EUROPE", "WBSC Europe Baseball Competitions", "international_regional", "senior", "mixed", "Europe",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", True,
        phase_type="tournament", rules_profile="wbsc_europe", discovery_priority=30,
        discovery_url="https://www.wbsceurope.org/",
        starter_source_id="wbsc_europe_baseball",
    ),

    # Additional high-coverage collegiate, youth, women's and amateur scope.
    CompetitionSpec(
        "NCAA_D1_BASEBALL", "NCAA Division I Baseball", "collegiate", "college", "mixed", "United States",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="league_and_tournament", rules_profile="ncaa_baseball_d1", discovery_priority=12,
        discovery_url="https://www.ncaa.com/sports/baseball/d1",
        starter_source_id="ncaa_baseball_d1",
        notes="Official NCAA schedule/results are public; team/roster/starter PIT must be validated independently.",
    ),
    CompetitionSpec(
        "NCAA_D2_BASEBALL", "NCAA Division II Baseball", "collegiate", "college", "mixed", "United States",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="league_and_tournament", rules_profile="ncaa_baseball_d2", discovery_priority=13,
        discovery_url="https://www.ncaa.com/sports/baseball/d2",
        starter_source_id="ncaa_baseball_d2",
    ),
    CompetitionSpec(
        "NCAA_D3_BASEBALL", "NCAA Division III Baseball", "collegiate", "college", "mixed", "United States",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="league_and_tournament", rules_profile="ncaa_baseball_d3", discovery_priority=14,
        discovery_url="https://www.ncaa.com/sports/baseball/d3",
        starter_source_id="ncaa_baseball_d3",
    ),
    CompetitionSpec(
        "WBSC_U12", "WBSC U-12 Baseball", "international_youth", "U12", "mixed", "international",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="tournament", rules_profile="wbsc_age_group", discovery_priority=31,
        discovery_url="https://www.wbsc.org/",
        starter_source_id="wbsc_age_group_reports",
    ),
    CompetitionSpec(
        "WBSC_U15", "WBSC U-15 Baseball", "international_youth", "U15", "mixed", "international",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="tournament", rules_profile="wbsc_age_group", discovery_priority=32,
        discovery_url="https://www.wbsc.org/",
        starter_source_id="wbsc_age_group_reports",
    ),
    CompetitionSpec(
        "WBSC_WOMENS_BASEBALL", "WBSC Women's Baseball World Cup", "international_womens", "senior", "female", "international",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="tournament", rules_profile="wbsc_womens_baseball", discovery_priority=33,
        discovery_url="https://www.wbsc.org/",
        starter_source_id="wbsc_womens_baseball",
    ),
    CompetitionSpec(
        "JAPAN_WOMENS_HIGH_SCHOOL", "Japan Women's High School Baseball", "high_school", "U18", "female", "Japan",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="tournament", rules_profile="japan_womens_high_school", discovery_priority=34,
        discovery_url="https://www.baseballjapan.org/",
        starter_source_id="omyu_womens_high_school",
    ),
    CompetitionSpec(
        "LITTLE_LEAGUE_WORLD_SERIES", "Little League World Series", "youth", "youth", "mixed", "international",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="tournament", rules_profile="little_league", discovery_priority=35,
        discovery_url="https://www.littleleague.org/world-series/",
        starter_source_id="little_league_world_series",
    ),
    CompetitionSpec(
        "JABA_CORPORATE", "Japan Amateur Baseball Association Corporate Baseball", "amateur", "senior", "mixed", "Japan",
        "COMPETITION_DEFINED", "RESEARCH_ONLY", False,
        phase_type="league_and_tournament", rules_profile="jaba_corporate", discovery_priority=36,
        discovery_url="https://www.jaba.or.jp/",
        starter_source_id="jaba_official",
    ),

    # Senior national-team tournaments / non-league competition.
    CompetitionSpec("WBC", "World Baseball Classic", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="wbc"),
    CompetitionSpec("WBSC_PREMIER12", "WBSC Premier12", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="wbsc_senior"),
    CompetitionSpec("OLYMPICS_BASEBALL", "Olympic Baseball", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="olympic_baseball"),
    CompetitionSpec("ASIAN_GAMES_BASEBALL", "Asian Games Baseball", "international_senior", "senior", "mixed", "Asia", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="asian_games"),

    # Youth / age-group national-team competitions.
    CompetitionSpec("WBSC_U18", "WBSC U-18 Baseball World Cup", "international_youth", "U18", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="wbsc_age_group"),
    CompetitionSpec("WBSC_U23", "WBSC U-23 Baseball World Cup", "international_youth", "U23", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="wbsc_age_group"),

    # Japanese high-school baseball: qualifiers and national tournaments are separate phases.
    CompetitionSpec("KOSHIEN_SENBATSU", "National High School Baseball Invitational (Senbatsu)", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="japan_high_school"),
    CompetitionSpec("KOSHIEN_SUMMER", "National High School Baseball Championship (Summer Koshien)", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="japan_high_school"),
    CompetitionSpec("KOSHIEN_QUALIFIERS", "Japanese High School Baseball Qualifiers", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="qualifier", rules_profile="japan_high_school_qualifier"),

    # Collegiate / amateur competition.
    CompetitionSpec("JAPAN_UNIVERSITY_BASEBALL", "Japanese University Baseball", "collegiate", "university", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="league_and_tournament", rules_profile="japan_university"),
)


def get(competition_id: str) -> CompetitionSpec:
    for spec in COMPETITIONS:
        if spec.competition_id == competition_id:
            return spec
    raise KeyError(f"unknown competition_id: {competition_id}")


def registry() -> list[dict[str, object]]:
    return [asdict(spec) for spec in COMPETITIONS]


def production_eligible(competition_id: str) -> bool:
    return get(competition_id).status == "PRODUCTION_ELIGIBLE"
