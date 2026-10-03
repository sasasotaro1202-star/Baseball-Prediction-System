# Baseball-Prediction-System — Project Instructions

The detailed operating specification is maintained in `docs/PROJECT_SOURCE.md`.
Use `docs/OPERATING_GOVERNANCE.md` as the compact operational gate.

Current GitHub state and verified runtime evidence are authoritative. Preserve historical results and failure records.

Required loop:
MONITOR → DETECT → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS.

PIT must be fail-closed when historical availability cannot be proven. Random splits are prohibited. Missing values must not become zero. Cross-project research requires local PIT/OOS/holdout validation.

Reference adoption benchmark:
primary relative LogLoss improvement >= 3%;
auxiliary improvement >= 1%;
no worsening in >= 70% of evaluation periods;
newest holdout no worsening;
no material calibration degradation;
PIT violations = 0.

These are promotion gates/benchmarks, not guarantees of future performance.
