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

1. Permutation leakage test.
2. Slow-vs-vectorized PIT equivalence.
3. Live/historical feature parity.
4. Chronological calibration stability and shared calibration implementation.
5. Baseline and challenger gates.
6. Feature-block ablation.
7. Reproducibility and deterministic replay.
8. Synthetic/contract fixtures for calibration and target integrity.

## Acceptance principle

The objective is not the highest backtest score. The objective is the strongest defensible performance on genuinely unknown future games, with no leakage, reproducible inputs, explicit uncertainty, and safe failure when the data contract is not satisfied.

These verification ideas are informed by current open-source baseball forecasting projects that use strict walk-forward evaluation, calibration, permutation leakage checks, slow point-in-time recomputation, synthetic fixtures, and honest baseline/market comparisons. They are references for test design, not automatic model imports.
