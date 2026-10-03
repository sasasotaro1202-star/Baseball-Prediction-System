# Baseball-Prediction-System — Project Source Alignment Addendum 2026-10-03

## Verified current state
- main latest observed HEAD after re-check: ce0796a3394ecc367f6148dc5d53a5632da360e0.
- Recent observed changes harden typed dispatcher invariants, add/test shadow-horizon breakdown assembly, and record NPB postgame prediction experience.
- The existing `docs/PROJECT_SOURCE.md` remains the detailed source of truth for NPB/MLB, starter/roster PIT, Statcast restrictions, calibration, OOS/WFO, experience learning and failure taxonomy.

## Alignment rules
1. NPB 3-class HOME/DRAW/AWAY and MLB 2-class HOME/AWAY remain independent target contracts.
2. starter/lineup/bullpen/weather/availability state is cutoff-dependent; late confirmation is not historical PIT evidence.
3. score and win probability forecasts are evaluated separately.
4. shadow horizon artifacts preserve canonical `game_id` and avoid duplicate-revision weighting.
5. scope expansion stays staged: coverage → PIT → chronological OOS → robustness → holdout → release/operations.
6. Current GitHub code/tests/evidence override stale documentation. No workflow-green or artifact-exists shortcut is permitted for adoption.
