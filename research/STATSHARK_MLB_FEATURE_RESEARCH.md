# StatsHawk MLB feature research candidates

This document is a research specification, not a production feature list.

## Candidate signals

The candidate set is motivated by observed StatsHawk PBP/box-score fields and external MLB research/implementation patterns.

| Candidate | Construction | Required evidence | Leakage status |
|---|---|---|---|
| pitcher recent K% | prior PA strikeouts / prior PA | each prior game record available before cutoff | blocked unless availability is proven |
| pitcher recent walk rate | prior BB / BF or PA denominator | same | blocked unless availability is proven |
| pitcher recent HR rate | prior HR / BF or PA denominator | same | blocked unless availability is proven |
| pitcher velocity trend | recent pitch start_speed vs baseline | per-pitch historical availability | blocked unless availability is proven |
| pitch-mix entropy/share | prior pitch_type distribution | per-pitch historical availability | blocked unless availability is proven |
| hard-contact proxy | prior in-play launch_speed distribution when present | per-pitch historical availability | blocked unless availability is proven |
| batter/pitcher platoon context | batter bat_side × pitcher throws | lineup/starter announcement timestamps | blocked until both personnel timestamps pass |
| lineup exposure | batting-order position / lineup confirmation | lineup publication timestamp | blocked without publication-time evidence |

## Construction rule

All rolling / trailing features must use strict as-of semantics:

source_available_at < prediction_cutoff

and may not use the target game's realized events.

For historical replay, a feature row is invalid when source availability is unknown. No retrieval timestamp, page timestamp, final box score timestamp, or provider confirmation flag may be substituted for an explicit availability/publication time.

## Independent-game aggregation

When the same game has multiple observations or revisions, collapse them to one independent game before statistical fitting or evaluation. Do not count revisions as independent samples.

## Research protocol

1. Collect prior-game observations.
2. Verify event identity against the canonical MLB source.
3. Attach source provenance and availability timestamps.
4. Fail closed on unknown availability.
5. Construct features using only strict prior observations.
6. Evaluate chronological OOS and protected holdout.
7. Compare calibration and worst-regime behavior before any promotion consideration.

## Current status

No candidate in this file changes the Production Champion. No automatic promotion is allowed.
