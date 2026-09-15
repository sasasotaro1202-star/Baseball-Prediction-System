# Baseball Prediction & Research System — Formal Specification v1.3

## Scope
- Baseball only, with a unified competition registry covering **professional, international, youth, collegiate/amateur, and major tournament baseball**.
- Initial production leagues remain NPB + MLB, but the system architecture must support and progressively ingest: NPB, MLB, WBSC Premier12, World Baseball Classic, Olympics, Asian Games, U18, U23, other WBSC age-group/national-team competitions, Koshien (National High School Baseball Championship and related qualifying/national tournaments), university baseball, and other material domestic/international competitions where reliable data exists.
- Competition-specific models are allowed when the data-generating process differs materially; do not mix competitions blindly. Cross-competition transfer is permitted only when validated by chronological OOS evidence.
- Precision is prioritized over execution speed.
- No fabricated data, guessed missing values, or unverified claims that code/workflows were executed.
- Existing Baseball prediction logic remains the domain baseline; generic research infrastructure is additive.

## Competition registry and data policy
Every event must carry a stable `competition_id`, `competition_type`, `level`, `gender`, `season`, `country/region` where applicable, and source provenance. The registry must distinguish at minimum:
- Professional: NPB, MLB and other professional leagues when reliable PIT data is available.
- International senior: Olympics, Asian Games, World Baseball Classic, WBSC Premier12 and equivalent national-team tournaments.
- International youth: U18, U23 and other WBSC age-group competitions.
- Japanese amateur/high school: Koshien (Spring Senbatsu and Summer National Championship), prefectural qualifiers, and other nationally significant high-school tournaments.
- Japanese collegiate/amateur: university baseball and national/major intercollegiate competitions.
- Other recognized domestic/international competitions may be added through the same evidence and PIT gates.

A competition is **not production-eligible merely because it is listed in the registry**. It must pass source coverage, PIT availability, outcome integrity, starter/roster rules appropriate to that competition, feature completeness, model validation, calibration, and minimum-sample gates. Unsupported competitions remain explicitly `RESEARCH_ONLY` or `UNAVAILABLE`, never silently mixed into production.

## Prediction eligibility
A production prediction is eligible only when the game, competition, start time, required pregame personnel information appropriate to the competition, required data, PIT snapshot, roster/availability state, feature completeness, model and calibration artifact are valid. For competitions with starting-pitcher announcements, both starting pitchers must be confirmed with verifiable announcement timestamps. If either required starter is unconfirmed, exclude the game. For youth/amateur/national-team competitions where pitcher announcement conventions differ, use the competition-specific documented eligibility rule; never substitute post-game actuals. Missing critical data is not replaced by a guess.

## Accuracy targets
The research system's explicit target is **80%+ accuracy for each defined classification output where that metric is statistically and operationally meaningful**. This is a target, not a guaranteed achieved result. The system must never manufacture or overstate accuracy to satisfy the target.

For primary win prediction:
- NPB: target Accuracy >= 80% for the 3-class Home/Draw/Away outcome.
- MLB: target Accuracy >= 80% for the 2-class Home/Away outcome.
- Other competitions: define the target only after a sufficiently large, representative chronological OOS sample exists; do not borrow the NPB/MLB target without evidence.

For the separate NPB Top-Draw selection: target hit rate >= 80% is aspirational and must be reported independently from the all-game 3-class accuracy. Because a forced daily draw selection can be intrinsically difficult, the system must allow `NO_SELECTION` when the evidence does not justify a draw selection; it must not force a false 80% claim.

For score candidates and Low/High, accuracy must be defined explicitly by the evaluation contract rather than conflated with win accuracy. The system should target >= 80% only for metrics whose denominator and event definition make an 80% target meaningful; otherwise optimize and report the appropriate proper scoring rule, MAE, top-k hit rate, calibration, or other domain metric.

## PIT / provenance
Every research-critical observation must distinguish event time, source timestamp, retrieval time and source availability time. Retrieval time never proves historical availability. PIT filtering must fail closed when availability cannot be established. Missing, unavailable and unverifiable are distinct from TRUE_ZERO.

## Starter / lineup timestamps
Store starter identity plus announcement timestamp for both teams when the competition has starting-pitcher announcements. Store lineup/roster status and announcement timestamp separately from projected/actual lineup. A post-game actual starter/lineup must never be substituted into a historical pregame row.

## NPB outcome contract
NPB is always a three-class target: `HOME_WIN`, `DRAW`, `AWAY_WIN`. Production probabilities must be `home`, `draw`, `away` and sum to 1. In addition to normal outcome prediction, select one separate Top-Draw game per Japan-local calendar date by maximum draw probability when a draw selection is warranted; otherwise allow `NO_SELECTION`.

## MLB outcome contract
MLB is binary home/away for the primary win market. Both starting pitchers must still be confirmed for production output.

## Other competition outcome contract
International, youth, high-school, university, and other competitions must use a competition-specific outcome contract reflecting their actual rules. Do not assume NPB's draw semantics or MLB's binary semantics universally. Exhibition/non-competitive games, games with materially incomplete records, and competitions without reliable final outcomes are excluded from supervised production training.

## Score prediction
Return four score candidates with probabilities. Score candidates are the four highest-probability exact score cells for that specific game, selected directly from the full game-specific score distribution. They must not be altered to agree with Low/High. Score evaluation must be chronological OOS and never use target-game/post-game information.

## Low / High
For the production system's canonical Low/High output, use the fixed **6.5 boundary**: LOW = total runs <= 6, HIGH = total runs >= 7, with `P(LOW)=P(total<=6)` and `P(HIGH)=P(total>=7)`. This label is independent of the four exact-score candidates. If a sportsbook/market total line is available at the prediction cutoff, calculate the separate market-line Over/Under probabilities and preserve that line as its own field; never overwrite the canonical 6.5 contract. If a requested market line is not verifiable at the cutoff, output `UNVERIFIABLE` rather than inventing a threshold.

## Calibration
Calibration is a separate versioned artifact applied at the final probability boundary. Temperature calibration may be fitted only on development/calibration data. Final production probabilities must record the calibration version.

## Prediction ledger
Every eligible production prediction is immutable and records: prediction id, event id, competition id/type/level, cutoff, creation time, teams, confirmed starters/personnel where applicable, probabilities, score candidates, Low/High, line, confidence, volatility, model version, feature version, calibration version, Git commit and data snapshot id.

## Research lifecycle
`Development OOS -> Candidate Selection -> Candidate Lock -> Independent Holdout -> Baseline vs Candidate -> ADOPT / REJECT / HOLD / NO_CHANGE`.
Holdout data must never influence candidate selection. Promotion must pass win, calibration, score, Low/High and competition-specific gates. NPB must additionally protect Draw Recall and Draw Probability MAE.

## MLB research lifecycle
MLB uses the same lifecycle and chronological candidate replay. Until PIT historical score/market-line evidence and verifiable starter-announcement evidence are connected, promotion is fail-closed rather than comparing incomplete evidence.

## Multi-competition research lifecycle
Each newly added competition must first pass an isolated historical data/PIT validation phase, then chronological OOS, calibration, robustness, and independent holdout evaluation. Only validated competitions may graduate from `RESEARCH_ONLY` to `PRODUCTION_ELIGIBLE`. Competition-specific models should be preferred where rule, talent pool, sample size, scoring environment, or data availability makes a shared model unreliable.

## Post-game feedback
Final results are joined by stable event id. Unmatched predictions are not silently counted as misses. The audit computes classification, probability, score and segment errors and exposes weaknesses for the next research cycle.

## Stability / weakness discovery
Analyze performance by season, competition, league, team/national team, starter where applicable, bullpen where applicable, home/away, favorite/underdog and sample size where available. Confidence, probability, data quality and volatility are separate concepts.

## Candidate governance
Candidates are reproducible, versioned and registered. Failed candidates remain auditable. Infrastructure improvement is not treated as model-performance improvement.

## Execution policy
Heavy backtests / GitHub Actions are not run automatically merely because code was changed. Lightweight inspection/tests and heavy research execution are separate operations. Any executed run must be reported accurately.

## Accuracy-governance rule
The 80% target is a **promotion target, not a permission to overfit**. A candidate cannot be promoted merely because it reaches 80% on one convenient slice. It must satisfy chronological OOS, independent holdout, calibration, minimum sample-size, stability, and leakage gates. If 80% is not achieved, report the measured result and continue research rather than altering labels, thresholds, eligibility, or evaluation windows to manufacture the target.

## Canonical implementation direction
Keep the existing `baseball_backtest.py` as the domain engine while progressively wiring PIT snapshots, availability timestamps, market-line snapshots, calibration, production prediction logging, MLB lifecycle and post-game auditing around it. Do not copy another sport's model into Baseball. Add a competition registry and adapter layer so new competitions can be ingested without contaminating the existing NPB/MLB contracts.

## Required S-rank components
1. Complete PIT data foundation.
2. Starter/lineup announcement timestamps where applicable.
3. Production Prediction Runner and eligibility gate.
4. Calibration fully connected to final probabilities.
5. MLB Development OOS -> Lock -> Independent Holdout lifecycle.
6. Historical total-runs market-line snapshots.
7. Immutable Prediction Log.
8. Automatic result reconciliation and error/weakness analysis.
9. Unified competition registry covering professional, international, youth, high-school and collegiate baseball, with explicit `RESEARCH_ONLY` / `PRODUCTION_ELIGIBLE` states.
10. Competition-specific PIT/data adapters and isolated OOS/holdout promotion gates.
