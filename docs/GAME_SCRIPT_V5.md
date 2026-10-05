# Game-Script v5 Research

## Purpose

Game-Script v5 is a research-only NPB challenger. It models the game as a sequence of state transitions and generates a full-game score/outcome distribution with Monte Carlo simulation.

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

Historical PBP is not assumed to be historically publication-time verifiable. Therefore every v5 screening result is:

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

Passing tests or a successful Actions run does not imply adoption. Promotion requires the project-wide chronological OOS, calibration, PIT, robustness, reproducibility, and frozen-holdout gates against the incumbent. Until those gates are satisfied, v5 remains a Challenger Research artifact.


## Incremental current-season refresh

The daily workflow separates PBP provenance into:

- validation source fingerprint for 2022-2025 historical WFO inputs;
- current-context fingerprint for 2026 season inputs;
- relevant Game-Script code fingerprint.

When only the current-context fingerprint changes and the code plus historical validation fingerprint remain unchanged, the workflow does not repeat historical Monte Carlo WFO. It restores a compatible complete validation checkpoint, rebuilds the full validation state, warms the kernel and team-strength state through the latest pre-shadow 2026 games, and refreshes the current-month frozen shadow only.

A code or historical-validation change invalidates the checkpoint and forces a fresh chronological WFO. This prevents stale metrics or model state from being silently reused after logic/data changes.


## Redundant automation path

The existing `Baseball 24h Research Autopilot` also contains an auxiliary Game-Script v5 fallback lane. It checks for a recent successful dedicated v5 run; when one exists it records a no-op delegation result, avoiding duplicate expensive computation. When the dedicated lane is absent or stale, the fallback downloads and verifies the same PBP release assets and runs either:

`FULL_WFO_FALLBACK`

or, when a compatible complete checkpoint exists:

`SHADOW_REFRESH_FALLBACK`.

This lane is research-only, does not modify production configuration, and preserves its own evidence artifact.

## v5 state-integrity hardening

v5 rejects transitions that could create illegal simulated game paths. A top-to-bottom or bottom-to-next-top transition must reset to 0 outs and empty bases, while within-half visible outs may increase by at most two and may never decrease. Malformed or underspecified boundary states are excluded rather than repaired.

The dedicated research loop is scheduled four times per day (00:45, 06:45, 12:45, 18:45 UTC). Unchanged fingerprints still skip full WFO, so the higher cadence primarily detects refreshed current-season context without duplicating expensive historical evaluation.
