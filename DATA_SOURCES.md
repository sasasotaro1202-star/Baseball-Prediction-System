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
| MLB schedule/results | MLB Stats API | cached verified data | Starting pitchers must be confirmed before live prediction |

## Completion rules

- A season is not complete merely because schedule rows were written.
- Both starting pitchers must be identified and both starter game lines must be available for at least the configured coverage threshold (currently 70%).
- Existing checkpoints are re-enriched when prediction-critical fields are missing.
- A missing upstream source never causes the collector to invent a value.
- Partial seasons remain resumable across GitHub Actions runs.

## Current implementation

`npb_runtime_patch.py` applies the source hierarchy and stale-checkpoint repair immediately before `npb_multi_source.py` runs. The GitHub Actions workflow performs this automatically, so the user only needs to dispatch the workflow (or wait for the scheduled run).
