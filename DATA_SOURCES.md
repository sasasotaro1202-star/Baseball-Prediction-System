# Baseball data source hierarchy

The system does not force every feature through one provider. Each feature uses the strongest practical source and records missing data instead of guessing.

| Feature | Primary source | Fallback / validation | Rule |
|---|---|---|---|
| NPB schedule, game date/time, teams, final result | NPB.jp official | SPAIA schedule API | Official NPB is canonical where available |
| NPB starting pitchers | SPAIA game PBP | NPB.jp official game page | Never substitute an arbitrary pitcher |
| NPB pitcher game line | SPAIA pitcher-game API | none | Missing line keeps starter coverage below completion gate |
| NPB player game stats / lineup / play style | SPAIA game APIs | public NPB data repository for future bulk recovery | Target-game stats are only committed after prediction |
| NPB PBP | SPAIA game PBP | armstjc/Nippon-Baseball-Data-Repository bulk PBP | Chronological, game-level data only |
| Historical NPB season/player reference stats | NPB.jp official statistics where available | public NPB data repository | Used only according to publication date / backtest cutoff |
| Historical weather | Open-Meteo archive | none | Venue/date cache; no synthetic weather |
| MLB schedule/results | MLB Stats API | Yahoo! Sports Navi MLB schedule/game pages; cached verified data | MLB Stats API remains canonical for IDs/results; Yahoo is a secondary Japanese-language discovery/validation source |
| MLB probable starters | Yahoo! Sports Navi MLB schedule/game pages + MLB Stats API | cross-source agreement | Probable starter observation is useful for discovery, but is not by itself proof of historical official announcement time |
| MLB Japanese-language game detail / starter display | Yahoo! Sports Navi MLB | MLB Stats API | Useful operational cross-check; preserve retrieved_at and source URL |

## Yahoo! Sports Navi MLB policy

`https://baseball.yahoo.co.jp/mlb/` and its `/schedule/` and `/game/<game_id>/` pages are supported as a **secondary operational source**. The source currently exposes MLB schedule/result pages and displays probable starters; game pages also expose starting lineups and starting pitchers. It is therefore useful for current-game discovery, cross-validation, and Japanese-language output. citeturn0search1turn0search4

The system must **not** infer `announcement_at` from page retrieval time, crawl time, first observation time, or the mere presence of a probable starter. A prediction is production-eligible only when the starter announcement timestamp is independently PIT-verifiable and is at or before `prediction_cutoff`. This preserves the existing fail-closed rule.

## Completion rules

- A season is not complete merely because schedule rows were written.
- Both starting pitchers must be identified and both starter game lines must be available for at least the configured coverage threshold (currently 70%).
- Existing checkpoints are re-enriched when prediction-critical fields are missing.
- A missing upstream source never causes the collector to invent a value.
- Partial seasons remain resumable across GitHub Actions runs.
- Secondary sources may improve coverage and cross-validation, but cannot override a stronger canonical source without an explicit conflict-resolution rule.

## Current implementation

`npb_runtime_patch.py` applies the source hierarchy and stale-checkpoint repair immediately before `npb_multi_source.py` runs. The GitHub Actions workflow performs this automatically, so the user only needs to dispatch the workflow (or wait for the scheduled run).

For MLB, Yahoo! Sports Navi is treated as a secondary operational feed; the dedicated PIT announcement gate remains authoritative. Unsupported historical announcement evidence must stay `RESEARCH_ONLY`/ineligible rather than being silently promoted.
