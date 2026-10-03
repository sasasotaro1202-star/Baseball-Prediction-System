# Baseball-Prediction-System — Operating Governance

This repository follows `docs/PROJECT_SOURCE.md` as its detailed technical, research, PIT, OOS, model, operations, and self-improvement source of truth.

## Objective
Optimize future generalization, case-level correctness, probabilistic quality, calibration, predictability awareness, uncertainty quality, robustness, PIT integrity, information efficiency, operational reliability, recovery, and reproducibility.

## Evidence boundary
Current GitHub HEAD, verified code, configuration, tests, workflow execution, artifacts, registries, and measured results take precedence over stale narrative documents. Historical experiment, failure, holdout, and production records are immutable.

## Promotion boundary
Candidate selection is development-only. Promotion requires chronological OOS, leakage/meta-leakage audit, PIT validity, reproducibility, calibration, robustness, and a frozen holdout. The reference benchmark is primary relative LogLoss improvement >=3%, auxiliary improvement >=1%, >=70% of evaluation periods without worsening, newest holdout without worsening, no material calibration regression, and zero PIT violations. These are gates/benchmarks, not guarantees of future performance.

## Production boundary
NPB is HOME/DRAW/AWAY; MLB is HOME/AWAY. Score Top-4 and LOW/HIGH are independent target versions. Canonical LOW/HIGH is total runs <=6 versus >=7. Unknown or unverifiable critical PIT data is fail-closed.

## Failure boundary
Never manufacture metrics, convert missing to zero, treat retrieved time as historical availability, hide exceptions, treat skipped tests as passed, or call failed recovery successful. Safe fallback, deferral, or abstention is preferable to an unsupported prediction.

## Transfer boundary
Research ideas from the other prediction repositories may be discovered and abstracted, but Baseball production adoption always requires local compatibility, local PIT, local chronological OOS, local robustness, and local frozen-holdout evidence.

## Continuous loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY → NEXT RESEARCH.
