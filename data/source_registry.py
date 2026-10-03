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
    SourceSpec("dbv_dbl_official", "Germany-DBL", "schedule_results_stats", "https://www.baseball.de/", 1, False, "official public DBL schedules/results/statistics; historical PIT granularity requires validation"),
    SourceSpec("dbsv_2bundesliga", "Germany-2BL", "schedule_results_stats", "Deutscher Baseball- und Softballverband", 1, False, "official public 2nd Bundesliga statistics; historical PIT/PBP granularity requires validation"),
    SourceSpec("bbsv_youth_baseball", "Germany-Youth", "age_group_competitions", "BBSV youth baseball competitions", 1, False, "official age-class schedules/rules/results are public; detailed historical PIT/PBP requires validation"),
    SourceSpec("rfebs_baseball", "Spain-Baseball", "league_and_age_group_competitions", "https://www.rfebs.es/es/disciplines/baseball", 1, False, "official Spanish baseball competition calendar/results; structured PBP and historical PIT require validation"),
    SourceSpec("lpbc_colombia", "Colombia-LPBC", "league_results_statistics", "https://www.lpbcol.com.co/", 1, False, "official public live statistics/results; historical PIT and structured acquisition require validation"),
    SourceSpec("northwoods_league", "USA-Summer-Collegiate", "schedule_statistics", "Northwoods League", 1, False, "official public schedule/statistics; historical PIT/PBP granularity requires validation"),
    SourceSpec("west_coast_league", "USA-Summer-Collegiate", "schedule_statistics", "West Coast League", 1, False, "official public schedule/statistics; historical PIT/PBP granularity requires validation"),
    SourceSpec("icba_ontario", "Canada-ICBA", "multi_age_schedule_results", "Inter County Baseball Association Ontario", 1, False, "public 2026 schedule/results across 8U-22U; detailed PBP and historical PIT require validation"),
    SourceSpec("baseball_quebec", "Canada-BaseballQuebec", "multi_age_competitions", "Baseball Québec", 1, False, "official 2026 championships span 9U-21U and female/male divisions; detailed PBP/PIT require validation"),
    SourceSpec("lfbq_womens", "Canada-WomensBaseball", "womens_multi_age", "Ligue Féminine de Baseball du Québec", 1, False, "public 2026 standings/rules and match data; historical PIT/PBP depth require validation"),
    SourceSpec("bbf_senior_leagues", "GreatBritain-Baseball", "senior_leagues", "British Baseball Federation", 1, False, "official 2026 divisions 1-5 schedules/results/statistics portal; historical PIT/PBP depth requires validation"),
    SourceSpec("bbf_youth_u16_u18", "GreatBritain-Youth", "youth_u16_u18", "British Baseball Federation Youth", 1, False, "official 2026 U16 plus youth U12/U14 competitions; detailed PIT/PBP depth requires validation"),
    SourceSpec("paba_philippines", "Philippines-Baseball", "national_and_tournament_results", "Philippine Amateur Baseball Association", 1, False, "official national-team event scoreboard/results; league-level historical PBP/PIT not yet verified"),
    SourceSpec("asian_games_baseball", "AsianGames-Baseball", "international_event_results", "Aichi-Nagoya 2026 Asian Games baseball", 1, False, "official event results are public; prediction-time PBP/PIT provenance requires validation"),
    SourceSpec("baseball_australia_youth", "Australia-Youth", "u16_u18_tournament_stats", "Baseball Australia Youth Championships", 1, False, "official 2026 U16/U18 schedule, results and tournament stats are public; underlying scoring service and historical PIT require validation"),
    SourceSpec("baseball_australia_womens", "Australia-Womens", "womens_national_championships", "Baseball Australia Women's Championships", 1, False, "official 2026 championship results and team statistics are public; detailed PBP/PIT requires validation"),
    SourceSpec("baseball_australia_womens_u16", "Australia-WomensYouth", "u16_womens_baseball", "Baseball Australia Youth Women", 1, False, "official 2026 U16 women's championship results are public; detailed PBP/PIT requires validation"),
    SourceSpec("lbda_juvenil_puerto_rico", "PuertoRico-Youth", "juvenile_baseball_stats", "Liga de Béisbol Doble A Juvenil de Puerto Rico", 1, False, "official accumulated/current and prior-season statistics are public; historical PIT/PBP depth requires validation"),
    SourceSpec("cbbs_brazil_competitions", "Brazil-Baseball", "national_youth_and_club_competitions", "CBBS", 1, False, "official public schedules, results and championship reports; detailed PBP/PIT depends on event"),
    SourceSpec("iab_israel_baseball", "Israel-Baseball", "multi_age_leagues_and_rules", "Israel Association of Baseball", 1, False, "official public leagues from grades 1-12 and adults, plus league rules; historical PIT/PBP depth requires validation"),
    SourceSpec("liga_argentina_beisbol", "Argentina-LAB", "national_league_schedule_results_stats", "Liga Argentina de Béisbol", 1, False, "official public scoreboard/results/statistics; historical PIT and granular PBP require validation"),
    SourceSpec("kbo_pbp_huggingface_2023_2026", "KBO", "pitch_by_pitch_tracking_research", "https://huggingface.co/datasets/slothman3878/kbo_playbyplay", 1, False, "Public CC-BY-4.0 derived corpus; regular season 2023-2026 partial; historical PIT availability requires separate validation."),
    SourceSpec("kbo_pbp_naver_github", "KBO", "pitch_by_pitch_research", "https://github.com/slothman3878/kbo_pbp_naver_sports", 1, False, "Public parser/source project; validates against official season totals according to project documentation; PIT requires separate validation."),
    SourceSpec("koshien_ranking_open_data", "Japan-HighSchool", "tournament_history_and_scores", "https://github.com/yumo120921/koshien-ranking", 1, False, "Public national and prefectural tournament history; detailed score coverage varies by round/year; not sufficient alone for full PIT-safe game modeling."),
    SourceSpec("koshien_history_open_data", "Japan-HighSchool", "historical_tournament_context", "https://github.com/chiisagosha/opendata", 2, False, "Public CC-BY-4.0 historical tournament finalist data 1948-2022; context source only, not full game-level PIT."),
    SourceSpec("cpbl_public_api_repository", "CPBL", "game_player_pitch_research", "https://github.com/MarkHungBuddha/cpbl", 1, False, "Public API/source repository with seasons, games, player stats and strike-zone endpoints; historical PIT remains unverified."),
    SourceSpec("npb_player_stats_2015_2025", "NPB", "player_season_prior", "https://github.com/yasumorishima/npb-prediction", 2, False, "Public 2015-2025 player/standing aggregates; useful as historical prior source but not game-level PIT evidence."),
    SourceSpec("npb_official_game_schedule_context", "NPB", "game_schedule_venue_context", "https://npb.jp/bis/{year}/games/gm{date}.html", 1, True, "Official date-specific schedule; start time/venue are observable at retrieval, and retrieval/availability timestamps must be preserved."),
    SourceSpec("npb_official_standings_central", "NPB", "team_standings_context", "https://npb.jp/bis/eng/{year}/stats/std_c.html", 1, False, "Official Central League standings; current/future prediction may use the snapshot observed at cutoff, while historical OOS requires historical availability evidence."),
    SourceSpec("npb_official_standings_pacific", "NPB", "team_standings_context", "https://npb.jp/bis/eng/{year}/stats/std_p.html", 1, False, "Official Pacific League standings; current/future prediction may use the snapshot observed at cutoff, while historical OOS requires historical availability evidence."),
    SourceSpec("open_meteo_forecast", "NPB+MLB", "pregame_weather_forecast", "https://api.open-meteo.com/v1/forecast", 2, False, "Hourly forecast snapshot near first pitch; available_at is the actual request observation time, and the forecast must remain isolated from historical OOS unless archived PIT evidence exists."),
    SourceSpec("npb_official_player_index", "NPB", "active_player_identity", "https://npb.jp/bis/players/active/", 1, False, "Official active-player index; player identity mapping is safe only at retrieval time and historical OOS requires the historical availability boundary."),
    SourceSpec("npb_official_player_page", "NPB", "player_profile_and_season_stats", "https://npb.jp/bis/players/{player_id}.html", 1, False, "Official player page with profile and year-by-year batting/pitching records; do not treat current-page values as historical PIT evidence without archived timing."),
    SourceSpec("npb_official_roster_status", "NPB", "date_scoped_roster_and_registration_status", "https://npb.jp/announcement/roster/roster_{MMDD}.html", 1, False, "Official date-specific first-team registration/de-registration and player list; retrieval time is an observation boundary, not proof of historical publication time."),
    SourceSpec("npb_official_team_batting", "NPB", "team_player_batting_stats", "https://npb.jp/bis/{season}/stats/idb1_{team}.html", 3, False, "Official NPB team individual batting table; current aggregate snapshot only. Historical OOS requires the page as-of/availability boundary to be proven and current-page revisions must not flow backward."),
    SourceSpec("npb_official_team_pitching", "NPB", "team_player_pitching_stats", "https://npb.jp/bis/{season}/stats/idp1_{team}.html", 3, False, "Official NPB team individual pitching table; current aggregate snapshot only. Historical OOS requires the page as-of/availability boundary to be proven and current-page revisions must not flow backward."),
    SourceSpec("npb_official_team_fielding", "NPB", "team_player_fielding_stats", "https://npb.jp/bis/{season}/stats/idf1_{team}.html", 3, False, "Official NPB team individual fielding table; current aggregate snapshot only. Historical OOS requires the page as-of/availability boundary to be proven and current-page revisions must not flow backward."),

)


def registry() -> list[dict[str, Any]]:
    return [asdict(s) for s in SOURCES]


def resolve(league: str, feature: str) -> list[SourceSpec]:
    return [s for s in SOURCES if (s.league == league or s.league == "NPB+MLB") and s.feature == feature]

