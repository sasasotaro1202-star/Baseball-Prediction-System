# Baseball Prediction & Research System — Formal Specification v1.1

## Scope
- Baseball only: NPB + MLB.
- Precision is prioritized over execution speed.
- No fabricated data, guessed missing values, or unverified claims that code/workflows were executed.
- Existing Baseball prediction logic remains the domain baseline; generic research infrastructure is additive.

## Prediction eligibility
A production prediction is eligible only when the game, league, start time, both starting pitchers, required data, PIT snapshot, roster/availability state, feature completeness, model and calibration artifact are valid. If either starter is unconfirmed, exclude the game. Missing critical data is not replaced by a guess.

## PIT / provenance
Every research-critical observation must distinguish event time, source timestamp, retrieval time and source availability time. Retrieval time never proves historical availability. PIT filtering must fail closed when availability cannot be established. Missing, unavailable and unverifiable are distinct from TRUE_ZERO.

## Starter / lineup timestamps
Store starter identity plus announcement timestamp for both teams. Store lineup status and announcement timestamp separately from projected/actual lineup. A post-game actual starter/lineup must never be substituted into a historical pregame row.

## NPB outcome contract
NPB is always a three-class target: `HOME_WIN`, `DRAW`, `AWAY_WIN`. Production probabilities must be `home`, `draw`, `away` and sum to 1. In addition to normal outcome prediction, select one separate Top-Draw game per Japan-local calendar date by maximum draw probability when a draw selection is warranted; otherwise allow `NO_SELECTION`.

## MLB outcome contract
MLB is binary home/away for the primary win market. Both starting pitchers must still be confirmed for production output.

## Score prediction
Return four score candidates with probabilities. Score evaluation must be chronological OOS and never use target-game/post-game information.

## Low / High
Use the actual historical total-runs market line when available at the prediction cutoff. Example: 7.5 means Low = 7 or fewer and High = 8 or more. If the line is not verifiable at the cutoff, output `UNVERIFIABLE`; never invent a threshold.

## Calibration
Calibration is a separate versioned artifact applied at the final probability boundary. Temperature calibration may be fitted only on development/calibration data. Final production probabilities must record the calibration version.

## Prediction ledger
Every eligible production prediction is immutable and records: prediction id, event id, league, cutoff, creation time, teams, confirmed starters, probabilities, score candidates, Low/High, line, confidence, volatility, model version, feature version, calibration version, Git commit and data snapshot id.

## Research lifecycle
`Development OOS -> Candidate Selection -> Candidate Lock -> Independent Holdout -> Baseline vs Candidate -> ADOPT / REJECT / HOLD / NO_CHANGE`.
Holdout data must never influence candidate selection. Promotion must pass win, calibration, score, Low/High and league-specific gates. NPB must additionally protect Draw Recall and Draw Probability MAE.

## MLB research lifecycle
MLB uses the same lifecycle and chronological candidate replay. Until PIT historical score/market-line evidence is connected, promotion is fail-closed rather than comparing incomplete evidence.

## Post-game feedback
Final results are joined by stable event id. Unmatched predictions are not silently counted as misses. The audit computes classification, probability, score and segment errors and exposes weaknesses for the next research cycle.

## Stability / weakness discovery
Analyze performance by season, league, team, starter, bullpen, home/away, favorite/underdog and sample size where available. Confidence, probability, data quality and volatility are separate concepts.

## Candidate governance
Candidates are reproducible, versioned and registered. Failed candidates remain auditable. Infrastructure improvement is not treated as model-performance improvement.

## Execution policy
Heavy backtests / GitHub Actions are not run automatically merely because code was changed. Lightweight inspection/tests and heavy research execution are separate operations. Any executed run must be reported accurately.

## Canonical implementation direction
Keep the existing `baseball_backtest.py` as the domain engine while progressively wiring PIT snapshots, availability timestamps, market-line snapshots, calibration, production prediction logging, MLB lifecycle and post-game auditing around it. Do not copy another sport's model into Baseball.

## Required S-rank components
1. Complete PIT data foundation.
2. Starter/lineup announcement timestamps.
3. Production Prediction Runner and eligibility gate.
4. Calibration fully connected to final probabilities.
5. MLB Development OOS -> Lock -> Independent Holdout lifecycle.
6. Historical total-runs market-line snapshots.
7. Immutable Prediction Log.
8. Automatic result reconciliation and error/weakness analysis.
