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
