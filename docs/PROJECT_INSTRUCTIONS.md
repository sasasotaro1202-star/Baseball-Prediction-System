# Baseball-Prediction-System — Project Instructions

## Priority and authority
`docs/PROJECT_SOURCE.md` is the detailed technical/research/validation/operations source of truth. Current GitHub HEAD, code, config, tests, workflows, Actions, artifacts, registries and measured evidence override older conversation or documents; historical results, failures and holdouts are immutable.

## Mandatory operating loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → FROZEN HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY → NEXT RESEARCH.

## Baseball target contract
NPB = HOME/DRAW/AWAY 3-class. MLB = HOME/AWAY 2-class. Score targets are independent from win targets. LOW/HIGH is independently versioned as LOW ≤ 6 runs / HIGH ≥ 7 runs. Competition phase UNKNOWN must not be silently mapped.

## PIT and data safety
Preserve game_time, prediction_time, cutoff, available_at, published_at, retrieved_at and revision_time. retrieved_at does not prove historical availability. Starter-dependent evidence requires announcement timing and cutoff eligibility. PIT uncertainty is UNKNOWN/UNVERIFIABLE and fail-closed. Missing is not zero; identity mismatches, stale critical data and target mismatches require fallback, abstention or deferral.

## Validation and selection
Random split is prohibited. Use chronological WFO/OOS with separate candidate selection, final OOS and frozen holdout. LogLoss is primary for win probabilities; report Accuracy, Brier, ECE, class-wise calibration and case-level diagnostics. Candidate adoption requires PIT validity, reproducibility, robustness, calibration, newest holdout and operational safety; benchmark thresholds are not guarantees.

## Cross-project research
Use the five-repository set as a research pool: Baseball, BTC, 7-Sport, Soccer and Stock. Transfer mechanisms, never raw performance claims. Every transfer must pass LOCAL PIT → LOCAL OOS/WFO → ROBUSTNESS → LOCAL FROZEN HOLDOUT → SHADOW before production consideration.

## Automation and failure policy
No 