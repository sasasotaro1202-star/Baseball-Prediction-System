# Game-Script v4 Research

## Purpose

Game-Script v4 is a research-only NPB challenger. It models the game as a sequence of state transitions and generates a full-game score/outcome distribution with Monte Carlo simulation.

The intended state is:

- inning band and top/bottom half
- outs
- base occupancy
- score differential bucket
- ball/strike count

The transition kernel uses hierarchical backoff:

rich state → no-count → no-score-diff → coarse base/out → half inning → global.

The model uses recency-weighted team offense/defense strength as a weak conditioning factor and applies shrinkage so small samples fall back toward league behavior.

## PIT policy

Historical PBP is not assumed to be historically publication-time verifiable. Therefore every v4 screening result is:

`pit_status=UNVERIFIABLE_HISTORICAL_PBP_AVAILABILITY`

`production_eligible=false`

and:

`decision=HOLD_RESEARCH_ONLY`

No historical PBP transition result may be promoted into production OOS without an independent availability proof.

Pregame/live use must be implemented as a separate PIT layer. A retrieved timestamp is never treated as a substitute for historical availability.

## Evaluation

The evaluator is expanding chronological WFO:

1. development games build the initial kernel and team-strength state;
2. each validation game is forecast before its outcome is added;
3. after the game, only that game's observed transitions and realized score are added;
4. the frozen shadow phase does not update from outcomes.

There is no random split and no holdout tuning.

Metrics include LogLoss, Brier, Accuracy, ECE, Score MAE, Top-1/Top-4 exact-score rate, Low<=6 classification metrics, fallback rate, Monte Carlo uncertainty, and an entropy-based predictability proxy.

## Robustness and failure behavior

- Missing bases/count are not imputed to zero.
- Invalid score reconstruction discards the game.
- Simulation coverage below 99.5% fails closed.
- Unsupported rich states back off deterministically and report the fallback rate.
- Monte Carlo uses a deterministic per-game seed.
- Checkpoints are atomically written and resume only when the checkpoint configuration exactly matches the current run.

## Automation

`.github/workflows/npb_game_script_autoresearch.yml` runs daily and on relevant main-branch changes. It:

1. verifies the exact main SHA;
2. snapshots the upstream PBP release manifest;
3. compares Git/source fingerprints with the previous successful evidence;
4. skips unchanged full recomputation;
5. downloads only missing/changed PBP assets and verifies SHA-256 digests;
6. restores/resumes the WFO checkpoint when appropriate;
7. publishes research evidence as an artifact.

`.github/workflows/npb_game_script_watchdog.yml` runs every six hours. It detects scheduler-blocked and stale executions, cancels only genuinely stale runs, and dispatches one bounded recovery run when the loop has been inactive for too long.

The watchdog never retries deterministic failures indefinitely and never changes production configuration.

## Promotion

Passing tests or a successful Actions run does not imply adoption. Promotion requires the project-wide chronological OOS, calibration, PIT, robustness, reproducibility, and frozen-holdout gates against the incumbent. Until those gates are satisfied, v4 remains a Challenger Research artifact.
