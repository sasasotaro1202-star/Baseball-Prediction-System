# Opta-like Baseball Data Sources

Research inventory for free-first baseball prediction data. This document does
not grant production eligibility to any source.

## MLB

### Baseball Savant / Statcast — PRIMARY RESEARCH SOURCE
- Public Statcast Search CSV endpoint.
- Pitch-level, game-level, player-level, team-level and season-level queries.
- Useful variables include pitch type, release speed, release spin, movement,
  plate location, launch speed, launch angle, hit distance, expected stats,
  pitch count/context, sprint speed, bat tracking and fielding metrics.
- Current repository integration: `data/mlb_statcast.py`.
- Status: IMPLEMENTED / research-only.
- PIT: historical publication/availability timestamp is not proven; therefore
  `historical_backtest_eligible=false` and `production_eligible=false`.

### pybaseball
- Open-source client pattern for Baseball Savant CSV access.
- Useful as an implementation reference; production code does not depend on it.
- Status: REFERENCE.

## NPB

### NPB+ / Hawk-Eye / NPB DMP — TARGET HIGH-FIDELITY SOURCE
- NPB+ is an NPB-authorized service using Hawk-Eye tracking data.
- NPB states that player tracking/performance data and pitch-by-pitch
  tracking visualization are provided.
- Direct free historical API/CSV access is NOT verified.
- Status: RESEARCH / ACCESS INVESTIGATION.

### wocchi09/npb-data — FREE PUBLIC RESEARCH SOURCE
- Daily collection of Sports Navi/Yahoo Japan pitch-by-pitch data.
- Captures pitch type, pitch speed, pitch result, course coordinates/zone and
  plate-appearance result.
- Provides historical daily JSON files and an automated GitHub Actions collector.
- Status: HIGH-VALUE FREE CANDIDATE; licensing/source terms must still be respected.

### icefields/BaseballNpbFetch — FREE PUBLIC API CLIENT
- Uses the public SPAIA NPB API without an API key.
- Exposes schedules, standings, player sabermetrics, lineups, game details,
  pitch-by-pitch history and play-by-play.
- MIT licensed client.
- Status: HIGH-VALUE FREE ACQUISITION/REFERENCE SOURCE.

### armstjc/Nippon-Baseball-Data-Repository — FREE PUBLIC RELEASE DATA
- Public NPB repository with PBP, schedules, rosters and player/game data.
- The current baseball system already consumes its PBP releases.
- Status: EXISTING CORE SOURCE.

### DataStadium
- Commercial distribution route for NPB DMP/tracking data.
- Useful for future investigation, but not part of the free-first production path.
- Status: COMMERCIAL / NO AUTO-USE.

## Research policy

A source can move from REFERENCE/SHADOW to OOS only after:
1. explicit historical availability/PIT evidence,
2. schema and source-integrity checks,
3. chronological OOS evaluation,
4. calibration/robustness validation,
5. no production regression,
6. reproducible artifacts.

Unknown historical availability remains FAIL-CLOSED.


## New findings — 2026-09-27

### MiLB Statcast / Baseball Savant
Baseball Savant provides a separate Minor League Statcast Search. Tracking coverage currently includes all Triple-A games from 2023 onward, Pacific Coast League and Charlotte home games from 2022, and Florida State League games from 2021. The search exposes pitch velocity, spin, movement, whiff, launch metrics, xwOBA/xSLG and related fields.
Source: https://baseballsavant.mlb.com/statcast-search-minors
Status: FREE PUBLIC / HIGH VALUE / coverage-dependent.

### NCAA Baseball — SportsDataverse
sportsdataverse/baseballr-data publishes NCAA schedules and play-by-play to GitHub release assets and maintains a daily collection workflow from stats.ncaa.org. 2026 repository metadata shows 39 NCAA PBP season assets and 59 schedule assets in the associated release inventory.
Source: https://github.com/sportsdataverse/baseballr-data
Status: FREE PUBLIC RELEASE DATA / HIGH VALUE / historical PIT must be audited.

### Retrosheet
Retrosheet provides event-file PBP, parsed PBP, game info, team/player logs, batting, pitching and fielding data. The current release describes complete AL/NL PBP coverage through 2025 and a parsed PBP corpus of over 208,000 games.
Source: https://www.retrosheet.org/
Status: FREE HISTORICAL RESEARCH DATA / very high historical value / not a direct PIT source.

### KBO — TrackMan / tracking
KBO documents league-wide automatic ball-strike tracking and public broadcast visualization based on TrackMan tracking. Publicly shown metrics include pitch type, velocity, RPM, batted-ball speed, launch angle and distance. A free public historical machine-readable feed of the tracking layer has not been verified.
Source: https://www.koreabaseball.com/Kbo/League/GameManage2026.aspx
Status: HIGH-VALUE TARGET / acquisition unresolved.

### KBO official statistics
Public KBO pages expose standings, hitter/pitcher records and player information. Independent open implementations demonstrate repeatable collection from the official site.
Status: FREE PUBLIC WEBSITE / acquisition and terms require audit before production use.

### CPBL
The open-source Rebas API project documents CPBL season, game, pitch-statistics and plate-appearance endpoints, plus winter-league (AWB), postseason and minor-league season identifiers. Some endpoints use authentication and the upstream service terms must be respected.
Source: https://github.com/MarkHungBuddha/cpbl
Status: RESEARCH CANDIDATE / potentially broad multi-competition coverage / access must be verified.

### Mexican Baseball League (LMB)
The official LMB site exposes current standings and player batting/pitching statistics. Open-source analysis implementations also exist, but a stable free public PBP API has not been verified.
Source: https://lmb.com.mx/
Status: FREE PUBLIC STATS / PBP API unresolved.

### Australian Baseball League (ABL)
Baseball Australia publishes ABL game notes and statistics, including player/team performance and historical season materials. Structured free PBP/tracking API access has not been verified.
Source: https://baseball.com.au/
Status: FREE PUBLIC MATERIALS / structured acquisition unresolved.

### WBSC / MyWBSC / international tournaments
WBSC operates MyWBSC as a digital statistics/live-scoring platform used by member federations and leagues. Public live-stat availability exists in some competitions, but a universal bulk API or historical PIT-safe feed is not verified.
Source: https://my.wbsc.org/
Status: INTERNATIONAL COVERAGE TARGET / acquisition unresolved.

## Discovery priority
Priority is now:
1. MLB Statcast + MiLB Statcast
2. NCAA SportsDataverse PBP
3. CPBL Rebas / AWB
4. KBO official stats + tracking access
5. NPB Hawk-Eye/DMP access
6. WBSC tournament feeds
7. LMB / ABL
8. Historical Retrosheet/Lahman layers for long-horizon modeling

No source moves into production solely because a public website exposes a metric. Production/OOS eligibility still requires schema integrity, PIT availability evidence, chronological validation, calibration and robustness evidence.


## Youth / high-school / age-group expansion — 2026-09-27

### 一球速報.com / OmyuTech — CROSS-AGE DISCOVERY SOURCE
The OmyuTech platform currently separates baseball into high school, university,
junior, JABA, amateur/softball, women's baseball, independent leagues and
regional categories. Its 2026 pages show:
- High school: prefectural, regional and national tournaments, with game,
  box-score, text/PBP, starting-information and pitch-level pages.
- Junior: Giants Cup and other junior tournaments; Little Senior, Boys, Young
  League and Pony organizations are represented.
- Women's baseball: national high-school women's tournaments and national
  junior women's tournaments.
- Youth/other: junior organizations can expose member, season and game histories.
This makes OmyuTech unusually valuable as a discovery layer for competitions
that are not covered by MLB/NPB-style professional feeds.
Sources:
https://baseball.omyutech.com/HomePageMain.action
https://baseball.omyutech.com/CupHomePageSokuhou.action?gameId=20261175013
https://baseball.omyutech.com/CupHomePageSeiseki.action?gameId=20260248306
https://baseball.omyutech.com/leagueCup.action?leagueId=70
Status: HIGH-VALUE FREE PUBLIC WEB RESEARCH CANDIDATE; scraping/API access,
terms and PIT availability require explicit validation.

### 日本高校野球連盟 (JHBF)
The official JHBF site publishes national high-school tournament schedules and
game results, including inning-by-inning score lines, game times, venues and
attendance for the Summer Koshien and related competitions.
Source:
https://jhbf.or.jp/sensyuken/2026/schedule/
Status: AUTHORITATIVE FREE PUBLIC RESULTS / strong target-label and schedule
source; pitch-level and historical PIT availability are not guaranteed.

### 高校野球 — OmyuTech detail depth
A 2026 Koshien box-score page exposes player batting rows, pitch counts per
plate appearance, individual pitch sequence/result text, pitch type and pitch
speed in km/h, along with game-level box score. This is substantially richer
than final-score-only sources.
Source:
https://baseball.omyutech.com/CupHomePageSeiseki.action?gameId=20261175013
Status: HIGH-VALUE RESEARCH CANDIDATE.

### 中学硬式 — Giants Cup / Boys + Little Senior
The 2026 Giants Cup page contains games between organizations such as Boys and
Little Senior. The box-score layer contains player-level batting and pitch
sequence details. This can create a cross-organization U-15-ish player/game
research corpus.
Source:
https://baseball.omyutech.com/CupHomePageSeiseki.action?gameId=20260248306
Status: HIGH-VALUE RESEARCH CANDIDATE.

### Young League / Pony / Little Senior
OmyuTech indexes multiple years of Young League competitions, Pony competitions,
and Little Senior team histories. Current 2026 examples include Young League
Junior Championship, Young League Championship, Pony national championships,
and Little Senior tournament histories.
Sources:
https://baseball.omyutech.com/leagueCup.action?leagueId=70
https://baseball.omyutech.com/teamGames.action?teamId=89725
Status: HIGH-VALUE DISCOVERY SOURCE; coverage varies by organizer and event.

### 学童 (elementary-age baseball)
OmyuTech indexes national and prefectural elementary-school baseball
competitions, including 2026 national tournaments. These pages include
tournament brackets, schedules/results and, where provided by the organizer,
live-score/game pages.
Sources:
https://baseball.omyutech.com/CupHomePageTournament.action?cupId=20260000894
https://baseball.omyutech.com/CupHomePageMain.action?cupId=20260010134
Status: FREE PUBLIC RESEARCH CANDIDATE; player identity/PIT quality likely more
variable than high-school/elite competitions.

### Women's high-school / junior
OmyuTech indexes 2026 national high-school women's tournaments and national
junior women's tournaments. Its junior women's pages can expose player,
batting, pitch-sequence and live text details.
Sources:
https://baseball.omyutech.com/teamGames.action?teamId=104226
https://baseball.omyutech.com/CupHomePageTextLive.action?gameId=20268716598
Status: HIGH-VALUE RESEARCH CANDIDATE.

### Samurai Japan age-group teams
The Japan national baseball team site publishes U-18, U-15 and other age-group
rosters, tournament schedules/results and detailed game tables. The 2026 U-15
World Cup and 2026 U-18 Asian Championship are current examples.
Sources:
https://www.japan-baseball.jp/jp/team/15u/2026/worldcup/overview.html
https://www.japan-baseball.jp/jp/team/18u/2026/asianchampionship/overview.html
Status: AUTHORITATIVE FREE PUBLIC TOURNAMENT DATA; ideal for international
age-group labels/results, but bulk PIT/tracking API remains unverified.

### WBSC age-group competitions
WBSC publishes official tournament statistics in structured PDF reports, including
U-12, U-18 and U-23 batting/pitching tables. These are useful for historical
team/player features and competition normalization, although they are not an
Opta-like universal real-time API.
Status: AUTHORITATIVE FREE HISTORICAL/TOURNAMENT REPORTS.

## Age/competition architecture candidate

The system can now discover and normalize:
PRO: NPB / MLB / KBO / CPBL / LMB / ABL / independent leagues
AMATEUR: NCAA / JABA / university / regional amateur
HIGH SCHOOL: Japan prefectural -> regional -> Koshien; women's high school
JUNIOR: U-15, Giants Cup, Little Senior, Boys, Young, Pony
ELEMENTARY: 学童
INTERNATIONAL AGE GROUP: WBSC U-12 / U-15 / U-18 / U-23 and national teams

The same target schema should not assume identical rules. Each competition must
carry rule metadata such as innings, tie handling, extra-inning format, mercy/run
rule, mound/distance conventions, designated hitter usage, and age class before
cross-competition modeling.


## Japanese Independent Leagues — 2026-09-27

### 四国アイランドリーグplus
- Official public data site exposes schedule/current games, standings, head-to-head
  results, batting and pitching statistics.
- Current season data is publicly viewable; historical prediction-time
  availability still requires PIT reconstruction before OOS use.
- Source: https://data.iblj.co.jp/

### ルートインBCリーグ
- Official data site exposes standings, head-to-head results, batting and
  pitching statistics for multiple seasons.
- Source: https://www.bc-l-data.jp/

### 一球速報.com / OmyuTech — independent leagues
- OmyuTech indexes Japanese independent leagues including the BC League and
  Shikoku Island League and exposes team, schedule and competition pages.
- Machine-access method, terms and historical PIT remain verification gates.
- Source: https://baseball.omyutech.com/

### Yahoo! Sports independent leagues
- Public independent-league player statistics provide a secondary cross-check
  for Japanese independent-league performance.
- Source: https://baseball.yahoo.co.jp/ipbl/stats/

Status: RESEARCH / HIGH-VALUE DOMESTIC INDEPENDENT-LEAGUE COVERAGE.

## Universal scope application — 2026-09-27

All currently registered baseball sources are mapped to an explicit competition
scope in:
- `research/competition_catalog.py`
- `research/universal_source_matrix.py`

The current catalog covers professional, college, high-school, junior,
elementary and international age-group scopes. Source capabilities are mapped
to canonical signals such as schedule identity, starters, lineups, batting,
pitching, fielding, bullpen, PBP, tracking, weather, news context and
tournament rules.

Application is additive and fail-closed:
- a source is applied only to scopes for which it is explicitly registered;
- missing information remains missing and is never converted to zero;
- historical research requires explicit `available_at <= prediction_time`;
- unknown PIT status is rejected;
- rule differences remain competition metadata rather than being silently
  normalized away.

This is the application layer requested for the collected source inventory.
It is research-only until each source passes its own PIT, data-quality,
chronological OOS, calibration and robustness gates.


## Additional global and development-league sources — 2026-09-27

### Latin American winter leagues
- MLB's winter-league statistics layer currently exposes LIDOM, LVBP, LMP and
  LBPRC statistical views.
- LVBP also has a public live-statistics/standings portal.
- LBPRC exposes public player batting/pitching statistics.
- These sources are valuable for winter-player form, roster context and
  cross-league player priors; historical PIT remains a separate requirement.
Sources:
https://www.mlb.com/ligas-invernales/stats/team
https://stats.lvbp.com/
https://www.ligapr.com/estadisticas/jugadores
Status: HIGH-VALUE PUBLIC STATISTICS / research-first.

### World Baseball Classic
- Official MLB WBC statistics expose team and player batting/pitching views
  for the 2026 tournament and prior editions.
- Useful as an international tournament domain and for country/roster
  normalization, but tournament-specific PIT and PBP granularity still need
  explicit validation.
Source:
https://www.mlb.com/world-baseball-classic/stats/team
Status: HIGH-VALUE INTERNATIONAL TOURNAMENT SOURCE.

### Little League World Series
- Little League International publishes official schedules, brackets and
  game recaps for the 2026 LLBWS.
- This adds a distinct youth domain that should not be mixed with high-school
  or junior-hardball rules without explicit competition metadata.
Sources:
https://www.littleleague.org/world-series/2026/llbws/tournaments/world-series/
https://www.littleleague.org/world-series/2026/llbws/bracket/
Status: AUTHORITATIVE YOUTH RESULTS / PBP depth varies.

### Cape Cod Baseball League
- The 2026 CCBL publishes official schedule plus player/team batting and
  pitching statistics.
- This is an important college-summer bridge because it concentrates
  collegiate players in a separate competition environment.
Sources:
https://www.capecodleague.com/about/schedule
https://www.capecodleague.com/stats
Status: HIGH-VALUE COLLEGE-SUMMER SOURCE.

### European baseball
- WBSC Europe provides competition/rule documentation across senior and
  youth categories such as U12, U15, U18 and U23.
- Czech Baseball Association's 2026 Extraliga pages expose schedules,
  results, standings and player statistics, with historical season selectors
  on team-stat pages.
- KNBSB exposes Dutch competition schedules/results/statistics and youth
  competition calendars and rules.
Sources:
https://www.wbsceurope.org/
https://m.baseball.cz/soutez/prehled
https://skokani.baseball.cz/statistiky
https://www.knbsb.nl/competities/
Status: HIGH-VALUE INTERNATIONAL / EUROPEAN RESEARCH SOURCES.

### Biomechanics research prior
OpenBiomechanics provides public processed biomechanics datasets for pitching,
hitting and high-performance assessments. The current snapshot is a small
research cohort and is not itself a game-time feed; it can only be used as a
representation/pretraining or player-prior candidate after explicit linkage
and temporal validity checks.
Source:
https://github.com/drivelineresearch/openbiomechanics
Status: RESEARCH-ONLY / NOT PREDICTION-TIME EVIDENCE.

## Newly registered application scopes
The universal catalog now additionally includes:
- WBC
- WBSC Women's Baseball
- Little League World Series
- Cape Cod Baseball League
- WBSC Europe
- Netherlands Youth Baseball
- Czech Baseball Competitions
- LIDOM / LVBP / LBPRC / LMP
- Japanese JABA corporate/amateur baseball

Every newly added source is required to have:
source registration → capability mapping → scope binding → PIT gate → OOS/calibration/robustness gate
before any production eligibility.


## Newly discovered development and regional layers — 2026-09-27

### MiLB public repository
The public `armstjc/milb-data-repository` contains schedules, season batting/pitching,
player-game statistics and PBP for AAA, AA, A+, A and Rookie levels. Its PBP guide
exposes play timestamps, pitch type and release speed, making this a high-value
development-layer source.
Source:
https://github.com/armstjc/milb-data-repository
Status: HIGH-VALUE RESEARCH SOURCE; historical publication/PIT boundary remains separate.

### CPBL advanced tracking research implementation
The public `lin-junyou/cpbl-savant-py-app` documents extraction from the CPBL
advanced-data site, including player records, per-game tables and TrackMan
pitch-level fields. The current public snapshot is explicitly a 2026-season
dataset, so it should be treated as a current/recent research layer rather than
a historical corpus unless older snapshots are proven.
Source:
https://github.com/lin-junyou/cpbl-savant-py-app
Status: HIGH-VALUE RESEARCH CANDIDATE; historical coverage limited/unverified.

### KBO public collector
The `kbo-data-portal/collector` project supports game, schedule and player
collection and documents season/stage filters, including a stated 1982-present
range. This is a useful independent acquisition path for KBO validation and
source redundancy.
Source:
https://github.com/kbo-data-portal/collector
Status: RESEARCH CANDIDATE; source terms and prediction-time availability require audit.

### Germany
The Deutsche Baseball Liga official site publishes schedules, boxscore links,
standings and statistics for the 2026 season; DBV/BBSV also publish second-tier
and youth age-class schedules, rules and historical archives.
Sources:
https://www.baseball.de/
https://www.baseball-softball.de/spielbetrieb/2-baseball-bundesliga/statistiken-2-baseball-bundesliga-2/
https://www.bbsv.de/spielbetrieb/spielklassen/
Status: HIGH-VALUE EUROPEAN SOURCE FAMILY.

### Spain
RFEBS publishes 2026 senior leagues and multiple youth competitions including
U12, U15 and U18 events, both club and autonomous-selection formats.
Source:
https://www.rfebs.es/es/disciplines/baseball
Status: HIGH-VALUE MULTI-AGE EUROPEAN SOURCE.

### France
The French federation publishes 2026 Division 1 competition news and directs
users to schedules, results and statistics in its competition system.
Source:
https://ffbs.fr/baseball/d1-baseball/
Status: HIGH-VALUE EUROPEAN SOURCE; structured PBP/PIT depth still requires validation.

### Colombia
The Liga Profesional de Béisbol Colombiana publishes schedule, live statistics,
results and standings through its official site.
Sources:
https://www.lpbcol.com.co/calendario/
https://www.lpbcol.com.co/transmisiones/
https://www.lpbcol.com.co/posiciones/
Status: HIGH-VALUE LATIN-AMERICAN RESEARCH SOURCE; historical PIT remains unverified.

### US summer collegiate
Northwoods League and West Coast League provide dedicated 2026 schedules and
statistics for summer collegiate baseball, expanding the collegiate development
environment beyond the NCAA regular season.
Sources:
https://northwoodsleague.com/scorebook/statistics/
https://westcoastleague.com/
Status: HIGH-VALUE COLLEGE-SUMMER SOURCE FAMILY.

### WBC player-prior dataset
A public 2026 WBC scouting dataset packages player-level Statcast-derived
regular-season information and roster/country metadata. It can be useful for
pre-event country/player priors but must not be treated as event-time WBC evidence.
Source:
https://github.com/yasumorishima/kaggle-datasets
Status: RESEARCH-ONLY DERIVED PRIOR.

## Discovery Frontier — still unverified
The current system should continue searching for machine-readable, public or
free-first sources for:
- Canadian Intercounty and provincial baseball
- Latin American summer/provincial circuits beyond the registered winter leagues
- Cuban Serie Nacional
- Nicaragua LBPN
- Panama Probeis
- Chinese mainland professional/amateur baseball
- Italy Serie A, Czech lower divisions, German lower divisions below 2BL
- Japanese regional university leagues and local club competitions
- U18/U15/U12 domestic leagues in Europe and Latin America
- women’s domestic leagues outside the currently registered Japan/WBSC scopes
- African and Middle-Eastern baseball competitions
- additional collegiate summer leagues in the US/Canada

These remain Discovery Targets until public data depth, terms, provenance and
prediction-time availability can be established.


## Canada / Great Britain / Australia / Puerto Rico multi-age additions — 2026-09-27

### Ontario Inter County Baseball Association
The 2026 season catalog contains 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U,
16U, 18U and 22U classes with multiple divisions, providing unusually broad
longitudinal youth coverage.
Source: https://icbabaseball.ca/Seasons/Current/
Status: HIGH-VALUE MULTI-AGE PUBLIC COMPETITION SOURCE.

### Baseball Québec / Québec Women's Baseball
Baseball Québec's 2026 championship framework spans 9U through 18U and 21U,
with both male and female divisions. The Québec women's league publishes 2026
rules, standings and match-data workflows.
Sources:
https://www.baseballquebec.com/fr/publication/federation/dates_des_championnats_2026.html
https://www.lfbq.ca/en/index.html
Status: HIGH-VALUE MULTI-AGE / WOMENS SOURCE FAMILY.

### Great Britain Baseball
The British Baseball Federation runs senior leagues through multiple divisions
and has active youth competitions including U12/U14 and a new U16 national
league in 2026.
Sources:
https://britishbaseball.org.uk/
https://britishbaseball.org.uk/bbf-announces-inaugural-u16-youth-national-baseball-league-for-2026/
Status: HIGH-VALUE SENIOR + YOUTH SOURCE FAMILY.

### Baseball Australia
The 2026 Australian Youth Championships provide U16 and U18 schedules, results,
team batting and pitching leaderboards. Baseball Australia also publishes
national senior and women's championship hubs.
Sources:
https://baseball.com.au/ayc2026/
https://baseball.com.au/2026nationals
Status: HIGH-VALUE MULTI-AGE / WOMENS SOURCE FAMILY.

### Puerto Rico Juvenile Double-A
The Liga de Béisbol Doble A Juvenil de Puerto Rico publishes current and prior
season accumulated statistics through its official portal.
Source:
https://www.lbdajpr.org/itinerario/estadisticas
Status: HIGH-VALUE YOUTH STATISTICS CANDIDATE.
