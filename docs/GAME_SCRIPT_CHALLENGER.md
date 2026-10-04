# Game Script Challenger

## Purpose

The Game Script module is a research-only challenger. It models NPB games as a chronological distribution of plate-appearance state transitions and then uses deterministic-seeded Monte Carlo simulation to derive:

- HOME / DRAW / AWAY probabilities
- home/away expected runs
- LOW (<=6) / HIGH (>=7) probabilities
- Top-4 exact-score candidates
- inning-level expected scoring
- simulation fallback/coverage signals

It does **not** replace the current production runtime and cannot self-promote.

## PIT boundary

The replay predicts a target game before allowing that game's PBP rows to update the learner. Games with the same start timestamp are predicted from the same frozen state snapshot.

The public PBP release used here does not carry sufficient historical source availability evidence to establish `available_at <= prediction_cutoff` for every row. Therefore the laboratory reports:

`pit_status = UNVERIFIABLE`

and:

`evidence_status = RESEARCH_ONLY_HOLD`

This is intentional. Temporal separation is tested, but temporal separation is not treated as a substitute for source-availability proof.

## State model

The transition key contains:

`offense team × inning bucket × half × outs × base occupancy × score-difference bucket`

The learner uses hierarchical backoff:

1. exact team/state
2. team/coarse state
3. league/exact state
4. league/coarse state
5. global transitions

This reduces variance when a specialist state has little data.

A transition stores:

`next inning × next half × next outs × next bases × runs scored × terminal flag`

The final plate appearance of each historical game contributes a terminal transition only after that game has passed through the prediction boundary.

## Automatic loop

`.github/workflows/baseball_game_script_research.yml` runs:

- every day for a recent-state experiment
- weekly for a deeper chronological experiment
- manually through `workflow_dispatch`

The runner has a bounded wall-clock budget. A budget exhaustion writes a checkpoint with:

- experiment fingerprint
- next chronological index
- completed prediction summaries

The workflow commits the checkpoint using a single-writer path. The next run rebuilds historical state up to that index and resumes deterministically.

Successful runs are stored under `results/game_script/runs/` and a compact `results/game_script/latest.json` is updated.

## Safety gates

The workflow refuses to treat the research as production evidence when:

- PIT availability is unverified
- the schema is unexpected
- probabilities or metrics are non-finite
- the module reports a status other than `EXECUTED`
- production eligibility is not explicitly `false`
- the decision is not `HOLD_RESEARCH_ONLY`

The current production runtime registry is not modified by this research loop.
