# Curated Interaction Layer

## Purpose

The candidate runner adds a small whitelist of pregame feature interactions. The goal is to let the model represent conditional effects that additive features cannot express while avoiding an uncontrolled all-pairs expansion.

## Interaction families

- starter quality × opponent strikeout/contact profile
- starter K-BB × opponent contact
- offense power × opposing starter HR/9
- offense walk rate × opposing starter BB/9
- offense strikeout rate × opposing starter K/9
- bullpen fatigue × expected run environment
- bullpen fatigue × starter quality gap
- lineup ISO × verified pregame wind
- total lineup power × verified pregame temperature
- recent form × run volatility

## Safety contract

The layer only consumes features already present in the prediction-time row. It does not read targets, future rows, post-game statistics, or market outcomes. Products are normalized by fixed scales and clipped to `[-5, 5]` to prevent a single extreme interaction from dominating.

## Promotion rule

The interaction candidate is **not production-adopted automatically**. It must beat the incumbent on chronological OOS using the repository's existing governance gates, and then pass independent holdout, calibration, stability, sample-size, and leakage checks. A single favorable slice is insufficient.

This follows the same discipline used by strong external forecasting projects: strict walk-forward evaluation, explicit leakage audits, calibration, and champion/challenger-style promotion rather than selecting the best-looking backtest. citeturn0search1turn0search7turn0search11
