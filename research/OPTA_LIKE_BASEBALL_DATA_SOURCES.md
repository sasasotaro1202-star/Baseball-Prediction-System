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
