"""Canonical source registry and provenance metadata for NPB/MLB and research sources."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    league: str
    feature: str
    endpoint: str
    priority: int
    critical: bool
    availability_rule: str


SOURCES = (
    SourceSpec("npb_schedule", "NPB", "schedule_result_identity", "https://npb.jp/", 1, True, "source availability must be <= cutoff"),
    SourceSpec("npb_starters", "NPB", "starting_pitchers", "https://spaia.jp/baseball/npb/api/flash_atbat_history", 1, True, "announcement timestamp <= cutoff"),
    SourceSpec("npb_lineups", "NPB", "lineups", "https://spaia.jp/baseball/npb/api/starting_members_for_flash", 1, False, "announcement timestamp <= cutoff"),
    SourceSpec("mlb_statsapi", "MLB", "schedule_result_identity", "https://statsapi.mlb.com/api/v1", 1, True, "source availability must be <= cutoff"),
    SourceSpec("mlb_starters", "MLB", "starting_pitchers", "https://statsapi.mlb.com/api/v1", 1, True, "confirmed pre-first-pitch information only"),
    SourceSpec("mlb_milb_statcast", "MiLB", "pitch_and_tracking_detail", "Baseball Savant Minor League Statcast", 1, False, "public data coverage varies by level/venue; historical availability boundary must be proven"),
    SourceSpec("ncaa_baseball_sportsdataverse", "NCAA", "schedule_and_pbp", "sportsdataverse/baseballr-data + sportsdataverse-data", 2, False, "released historical data; prediction-time publication availability must be separately proven"),
    SourceSpec("retrosheet", "MLB+Historical", "historical_pbp_and_boxscore", "https://www.retrosheet.org/", 2, False, "historical research source; prediction-time PIT availability is not inherently represented"),
    SourceSpec("kbo_official_tracking", "KBO", "tracking_detail", "KBO / TrackMan-based tracking + ABS", 1, False, "tracking system exists; public programmatic historical access is unverified"),
    SourceSpec("kbo_official_stats", "KBO", "game_and_player_stats", "https://www.koreabaseball.com/", 2, False, "public website data; scraping/access terms and historical PIT require verification"),
    SourceSpec("cpbl_rebas", "CPBL", "game_player_pitch_stats", "https://www.rebas.tw/", 2, False, "public endpoints documented; authentication/terms vary by endpoint; PIT must be proven"),
    SourceSpec("lmb_official", "LMB", "standings_and_player_stats", "https://lmb.com.mx/", 2, False, "official public statistics; API/history/PIT availability unverified"),
    SourceSpec("abl_official", "ABL", "league_and_player_stats", "https://baseball.com.au/", 2, False, "public official materials/statistics; structured API/PIT availability unverified"),
    SourceSpec("wbsc_mywbsc", "WBSC+International", "tournament_live_stats", "MyWBSC", 2, False, "live-stat infrastructure exists; bulk public API and historical PIT availability unverified"),
    SourceSpec("omyu_high_school", "Japan-HighSchool", "game_and_pitch_by_pitch", "一球速報.com / OmyuTech", 2, False, "public pages expose game, box score, PBP and pitch detail; machine-access method and historical PIT must be verified"),
    SourceSpec("omyu_junior", "Japan-Junior", "game_and_pitch_by_pitch", "一球速報.com / OmyuTech", 2, False, "covers junior hardball leagues and national tournaments; machine-access method and historical PIT must be verified"),
    SourceSpec("omyu_little_senior", "Japan-Junior", "game_and_boxscore", "一球速報.com / OmyuTech", 2, False, "Little Senior games/team records are publicly indexed; PIT/access terms require verification"),
    SourceSpec("omyu_boys", "Japan-Junior", "game_and_boxscore", "一球速報.com / OmyuTech", 2, False, "Boys League teams/games appear in the junior data layer; PIT/access terms require verification"),
    SourceSpec("omyu_young", "Japan-Junior", "game_and_boxscore", "一球速報.com / OmyuTech", 2, False, "Young League has multi-season tournament records and plate-appearance reporting; PIT/access terms require verification"),
    SourceSpec("omyu_pony", "Japan-Junior", "game_and_boxscore", "一球速報.com / OmyuTech", 2, False, "PONY junior hardball leagues and national tournaments are indexed; PIT/access terms require verification"),
    SourceSpec("omyu_youth_girls", "Japan-WomensYouth", "game_and_pitch_by_pitch", "一球速報.com / OmyuTech", 2, False, "national high-school girls and junior girls tournaments expose member, box-score and live/PBP pages; PIT/access terms require verification"),
    SourceSpec("jhbf_official", "Japan-HighSchool", "official_schedule_and_results", "https://jhbf.or.jp/", 1, False, "official tournament results are public; pitch-level/PIT availability is not guaranteed"),
    SourceSpec("samurai_youth", "Japan-U12-U15-U18", "national_team_results_and_rosters", "https://www.japan-baseball.jp/", 1, False, "official age-group tournament schedules, results, rosters and game tables are public; bulk historical PIT is unverified"),
    SourceSpec("statcast", "MLB", "pitch_level_and_pitcher_detail", "Baseball Savant / Statcast", 1, False, "publication/event information must be <= cutoff"),
    SourceSpec("npb_hawkeye_npbplus", "NPB", "tracking_data", "NPB+ / NPB DMP / Hawk-Eye", 1, False, "direct public API is not verified; every feature requires explicit available_at <= prediction_time"),
    SourceSpec("npb_public_spaia_pbp", "NPB", "pitch_level_public_research", "https://spaia.jp/baseball/npb/api/flash_atbat_history", 2, False, "current retrieval may be used only for current prediction; historical OOS requires archived available_at evidence"),
    SourceSpec("fangraphs", "MLB", "advanced_batting_pitching_reference", "FanGraphs", 2, False, "historical metric must be cutoff-safe"),
    SourceSpec("weather", "NPB+MLB", "weather", "Open-Meteo", 2, False, "archived observation/forecast snapshot only"),
    SourceSpec("x_api_recent_search", "NPB+MLB", "public_social_context_research", "https://api.x.com/2/tweets/search/recent", 3, False, "observed_available_at is the earliest safe availability boundary; historical availability is unproven"),
    SourceSpec("kbo_naver_pbp_public", "KBO", "pitch_by_pitch_public_research", "GitHub slothman3878/kbo_pbp_naver_sports + public dataset", 1, False, "source event/publication boundary must be proven before historical OOS"),
    SourceSpec("wocchi09_npb_data", "NPB", "public_pitch_by_pitch_research", "GitHub wocchi09/npb-data", 1, False, "public archive candidate; source terms and historical availability boundary require validation"),
    SourceSpec("armstjc_npb_repository", "NPB", "historical_pbp_and_boxscore", "GitHub armstjc/Nippon-Baseball-Data-Repository", 2, False, "historical release source; prediction-time PIT is not inherently represented"),
    SourceSpec("lahman", "MLB+Historical", "historical_season_and_player_reference", "Sean Lahman Baseball Database", 2, False, "historical reference; never treated as prediction-time evidence without PIT proof"),
    SourceSpec("omyu_university", "Japan-University", "game_and_pitch_by_pitch", "一球速報.com / OmyuTech", 2, False, "public pages vary by organizer; machine access, terms and PIT require validation"),
    SourceSpec("omyu_jaba", "Japan-Amateur", "game_and_pitch_by_pitch", "一球速報.com / OmyuTech", 2, False, "public pages vary by organizer; machine access, terms and PIT require validation"),
    SourceSpec("omyu_elementary", "Japan-Elementary", "game_and_boxscore", "一球速報.com / OmyuTech", 2, False, "coverage varies by organizer; player identity and PIT require validation"),
    SourceSpec("omyu_womens_high_school", "Japan-WomensHighSchool", "game_and_pitch_by_pitch", "一球速報.com / OmyuTech", 2, False, "public pages vary by organizer; machine access and PIT require validation"),
    SourceSpec("omyu_womens_junior", "Japan-WomensJunior", "game_and_pitch_by_pitch", "一球速報.com / OmyuTech", 2, False, "public pages vary by organizer; machine access and PIT require validation"),
    SourceSpec("samurai_u23", "Japan-U23", "national_team_results_and_rosters", "Japan Baseball / WBSC-related U23 competitions", 1, False, "official results are public; bulk historical PIT/PBP availability is unverified"),
    SourceSpec("wbsc_age_group_reports", "WBSC-U12-U15-U18-U23", "tournament_reports_and_stats", "WBSC official tournament reports", 1, False, "historical reports are public; prediction-time availability and bulk PBP must be proven"),
    SourceSpec("omyu_independent", "Japan-Independent", "game_and_pitch_by_pitch", "一球速報.com / OmyuTech", 1, False, "public pages vary by organizer; machine access, terms and PIT require validation"),
    SourceSpec("iblj_official_stats", "Japan-Independent", "league_and_player_stats", "https://data.iblj.co.jp/", 1, False, "official public statistics; historical availability boundary must be audited"),
    SourceSpec("bcl_official_stats", "Japan-Independent", "league_and_player_stats", "https://www.bc-l-data.jp/", 1, False, "official public statistics; historical availability boundary must be audited"),
    SourceSpec("yahoo_ipbl_stats", "Japan-Independent", "league_and_player_stats", "Yahoo! Sports 独立リーグ", 2, False, "public statistics; publication boundary and access terms require audit"),
    SourceSpec("lidom_mlb_winter", "LIDOM", "winter_league_stats", "MLB Winter Leagues / LIDOM", 1, False, "public stats are available; historical prediction-time publication boundary must be proven"),
    SourceSpec("lvbp_official", "LVBP", "league_and_player_stats", "https://stats.lvbp.com/", 1, False, "public league statistics/live pages; historical PIT and structured acquisition require audit"),
    SourceSpec("lbprc_official", "LBPRC", "league_and_player_stats", "https://www.ligapr.com/", 1, False, "public league statistics; historical PIT and structured acquisition require audit"),
    SourceSpec("lmp_mlb_winter", "LMP", "winter_league_stats", "MLB Winter Leagues / Liga Mexicana del Pacifico", 1, False, "public stats are available; historical prediction-time publication boundary must be proven"),
    SourceSpec("wbc_official_stats", "WBC", "international_tournament_stats", "MLB/WBC official statistics", 1, False, "official tournament stats are public; prediction-time availability and PBP granularity require audit"),
    SourceSpec("wbsc_international_events", "WBSC+International", "tournament_schedule_results_stats", "WBSC official events and reports", 1, False, "official schedules/results/reports are public; bulk PIT/PBP availability varies by event"),
    SourceSpec("wbsc_womens_baseball", "WBSC-WomensBaseball", "international_womens_baseball", "WBSC Women's Baseball World Cup", 1, False, "official event statistics are public; historical PIT/PBP depth varies by event"),
    SourceSpec("little_league_world_series", "LittleLeague", "youth_tournament_results", "Little League International", 1, False, "official schedules/brackets/results are public; detailed PBP and historical PIT require validation"),
    SourceSpec("cape_cod_league", "CCBL", "college_summer_stats", "Cape Cod Baseball League", 1, False, "official public schedule and batting/pitching statistics; historical PIT/source terms require audit"),
    SourceSpec("wbsc_europe_baseball", "WBSC-Europe", "european_baseball_competitions", "WBSC Europe", 1, False, "official competitions/rules/reports are public; bulk PBP/PIT availability varies"),
    SourceSpec("knbsb_baseball", "Netherlands-Baseball", "league_and_youth_competitions", "KNBSB", 2, False, "official competition schedules/results/stats and youth classes are public; structured PIT requires validation"),
    SourceSpec("openbiomechanics", "Cross-Level-Research", "biomechanics_player_prior", "Driveline OpenBiomechanics Project", 2, False, "public research dataset; not game-time evidence, small research cohort, player linkage/PIT required"),
    SourceSpec("jaba_official", "Japan-Amateur", "corporate_baseball_competitions", "https://www.jaba.or.jp/", 1, False, "official schedules/results are public; structured PBP and historical PIT require validation"),
    SourceSpec("big6_scorebook", "Japan-University", "university_league_records", "https://big6scorebook.jp/", 1, False, "official record room exposes schedules, results and player/team records; granular PIT requires validation"),
    SourceSpec("czech_baseball_assoc", "Czechia-Baseball", "league_and_youth_stats", "Czech Baseball Association / baseball.cz", 1, False, "official schedules/results/statistics are public; historical PIT and PBP granularity require validation"),
    SourceSpec("milb_data_repository", "MiLB", "schedule_pbp_player_stats", "GitHub armstjc/milb-data-repository", 1, False, "public release corpus; historical prediction-time availability must be separately proven"),
    SourceSpec("cpbl_savant_tracking", "CPBL", "trackman_pbp_tracking", "GitHub lin-junyou/cpbl-savant-py-app / stats.cpbl.com.tw", 1, False, "publicly retrievable research corpus; current 2026 scope and historical PIT must be validated"),
    SourceSpec("kbo_data_portal_collector", "KBO", "historical_game_schedule_player_stats", "GitHub kbo-data-portal/collector", 1, False, "public collector claims 1982-present; source terms and prediction-time availability require audit"),
    SourceSpec("ffbs_d1_official", "France-D1", "schedule_results_stats", "Fédération Française de Baseball et Softball", 1, False, "official public competition pages; structured PBP and historical PIT depth require validation"),
    SourceSpec("wbc_scouting_public_dataset", "WBC+PlayerPrior", "player_pitch_level_pre_event_prior", "GitHub yasumorishima/kaggle-datasets WBC 2026 scouting dataset", 2, False, "derived public dataset using MLB regular-season Statcast; use only as pre-event prior with explicit temporal boundary"),
)


def registry() -> list[dict[str, Any]]:
    return [asdict(s) for s in SOURCES]


def resolve(league: str, feature: str) -> list[SourceSpec]:
    return [s for s in SOURCES if (s.league == league or s.league == "NPB+MLB") and s.feature == feature]

