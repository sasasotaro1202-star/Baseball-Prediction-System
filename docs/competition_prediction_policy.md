# Competition-aware prediction policy

The baseball system separates evaluation and prediction policy at two levels:

1. League / product: NPB uses Home/Draw/Away while MLB uses Home/Away.
2. Competition / phase: regular season, interleague, postseason, tournament/qualifier, exhibition, and unknown are tracked by competition_key.

A competition does not become production-eligible merely by appearing in the registry.

## Prediction routing

The base model portfolio remains league-specific. On top of it, chronological OOS validation can apply a competition-specific calibration temperature:

- regular season / interleague: league_adaptive_ensemble + competition temperature
- postseason: postseason_shrunk_ensemble + competition temperature
- tournament / qualifier: tournament_shrunk_ensemble + competition temperature
- exhibition / unknown: league-level calibration only

The competition temperature is fitted only from validation predictions strictly earlier than the next walk-forward block. A small sample or unclassified competition falls back to the league-level temperature.

This is deliberately hierarchical rather than a hard model switch. It avoids fitting a fragile specialist from a tiny tournament sample while still allowing competition-specific calibration when evidence exists.

## Evaluation outputs

For each league, OOS results preserve:

- competition_key
- competition / stage / season type / game class
- prediction strategy id
- calibration scope
- Accuracy / LogLoss / Brier / ECE
- Low/High accuracy
- Score MAE
- Top-4 exact-score hit rate

Two files are produced:

- <league>_competition_metrics.csv
- <league>_competition_target_metrics.csv

The second file keeps Win, Low/High, and Exact Score as independent prediction targets.

## Fail-closed rules

Unknown competition identity is never silently mapped to a regular-season bucket. Research-only competitions remain blocked from production until their own PIT, OOS, calibration, robustness, provenance, and locked-holdout evidence passes.

The policy is research/validation infrastructure. It does not automatically promote a model.