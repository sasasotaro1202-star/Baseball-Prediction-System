# Baseball backtest audit

This file documents the verification requirements for the NPB backtest pipeline.

## Hard requirements

- Chronological walk-forward evaluation only.
- Target-game outcome and target-game performance must never enter pregame features.
- Starting pitchers must be resolved explicitly; never substitute arbitrary pitchers when unresolved.
- Existing checkpoints must never be overwritten with an empty dataset merely because a source temporarily returned no rows.
- Aggregate completion must reject unavailable years and enforce the configured starter-line coverage threshold.
- Player-game performance features for a target game may only be used after that game has been scored; target-game rows may be used for lineup identity/order only when identity is otherwise unavailable.
- Missing data must remain missing/neutral and reduce confidence rather than being guessed.

## Current known integration risks to verify

1. Collector starter fields must flow into the aggregate game table.
2. Per-game pitcher metrics must flow into prior-history starter features.
3. Target-game lineup identity/order must not be confused with target-game performance.
4. 1990-2018 historical schemas require explicit parser support; zero-row results must not be marked complete.
5. Backtest completion must be based on actual usable data, not only checkpoint file existence.

## Validation protocol

Run syntax compilation first, then a small chronological smoke test, then the full NPB-only backtest. Report coverage, starter coverage, Brier score, log loss, AUC, and any unavailable/missing years. Never report an unverified run as successful.
