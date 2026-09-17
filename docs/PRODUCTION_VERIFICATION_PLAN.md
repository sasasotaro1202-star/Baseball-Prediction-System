# Production Verification Plan

This document records verification work that should improve real-world robustness without weakening the production gates.

## Non-negotiable evaluation rules

- Walk-forward / chronological evaluation only.
- Prediction-time information must be point-in-time available at the prediction cutoff.
- Calibration fitting must use rows strictly earlier than the rows used to score the calibrator.
- Independent holdout periods are never used to choose features, thresholds, models, or hyperparameters.
- Missing or ambiguous starter-announcement timing is fail-closed for historical starter-dependent evaluation.
- Failed quality gates must remain failures; recovery may retry infrastructure/transient failures but may not convert deterministic failures into success.

## Verification layers to add or strengthen

1. **Permutation leakage test**
   - Shuffle targets in a deterministic offline fixture.
   - Run the same OOS pipeline.
   - Require performance to collapse toward the shuffled baseline within a pre-declared tolerance.

2. **Slow-vs-vectorized PIT equivalence**
   - Recompute a sampled set of historical features using an intentionally simple cutoff filter.
   - Compare against the production/vectorized feature builder.
   - Fail on meaningful numerical or row-level disagreement.

3. **Live/historical parity**
   - Where separate builders exist, run both against identical PIT fixtures.
   - Require equivalent feature values and identical missing-data decisions.

4. **Calibration stability**
   - Report Brier, log loss, reliability bins, sample counts, and uncertainty by chronological period.
   - Do not accept a calibrator merely because it improves the same rows used to fit it.

5. **Baseline and challenger gates**
   - Compare every candidate against simple chronological baselines.
   - Promote a challenger only when improvement survives OOS evaluation and does not introduce materially worse missing-data or stability behavior.

6. **Feature ablation**
   - Evaluate feature blocks rather than adding features by intuition.
   - Retain only blocks that improve true OOS behavior or provide a documented robustness benefit.

7. **Reproducibility**
   - Same immutable inputs, seed, code revision, and configuration should reproduce the same predictions/metrics.
   - Any nondeterminism should be explicit and tested.

## External design references

The verification philosophy is informed by current open-source baseball forecasting projects that use strict walk-forward evaluation, calibration, permutation leakage checks, slow point-in-time recomputation, synthetic fixtures, and honest baseline/market comparisons. These references are used for test design ideas only; their model choices are not imported automatically.

## Acceptance principle

The objective is not the highest backtest score. The objective is the strongest defensible performance on genuinely unknown future games, with no leakage, reproducible inputs, explicit uncertainty, and safe failure when the data contract is not satisfied.
