# PROJECT_INSTRUCTIONS — Baseball-Prediction-System

## Mission
Maximize future-game generalization, case-level correctness, calibration, predictability awareness, uncertainty quality, robustness, PIT integrity and operational reliability for NPB/MLB prediction.

## Every run
Re-check latest GitHub HEAD/default branch, code/config, tests, workflows, Actions, artifacts, registries, production/champion/challenger state, OOS/holdout evidence, failures and backlog. Prefer REUSE → REPAIR → INTEGRATE → TEST → VERIFY. Never rewrite historical evidence.

## PIT / time
Separate game_time, prediction_time, cutoff, source availability/publication, retrieval and revision. Starter, lineup, bullpen, weather, market and player-availability information is time-varying. Later confirmations or corrections cannot flow backward. Postgame/finalized same-game statistics are prohibited from pregame features.

## Target isolation
NPB uses HOME/DRAW/AWAY; MLB uses HOME/AWAY. Score distributions, exact-score candidates and LOW/HIGH are separate target contracts. Do not convert incompatible targets silently.

## Evaluation
Chronological WFO/OOS only; random temporal splits are prohibited. Candidate selection and final OOS are separate. Frozen holdout is final evidence only and must not be tuned.

## Models / routing
Maintain naive/class-frequency/Elo/logistic baselines. Complex models, starter experts, Statcast experts and specialist routing require incremental OOS value, adequate sample/folds, calibration evidence and robustness. Sparse scopes fallback to broader validated models.

## Calibration / uncertainty
Evaluate LogLoss, Brier, ECE and calibration parameters chronologically. Track model disagreement, starter uncertainty, source uncertainty, OOD and regime ambiguity separately from confidence.

## Reliability
Use checkpoint/resume, idempotency, bounded retry/backoff, watchdog/heartbeat, deterministic writes, artifact preservation, recovery and rollback. Never mask failure. Missing is never zero.

## Cost/security
Prefer verified free/OSS/local/cache. Unknown-cost or billing-risk services are not automatic dependencies. Never expose keys/tokens/secrets.

## Reconciliation metadata
Prediction-time competition/phase metadata captured by the production runner must survive postgame reconciliation into the experience ledger. Reconciliation may enrich missing metadata only from the immutable prediction snapshot or separately verified prediction-time evidence; it must not infer a phase from postgame knowledge or silently remap UNKNOWN.

## Completion
Green Actions or generated artifacts are execution evidence, not automatic performance verification. Completion requires tests, PIT/leakage, identity/scope audit, chronological OOS, calibration, ablation, robustness, frozen holdout, artifact integrity, reproducibility, recovery, release gate, monitoring and rollback.

## Loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY → NEXT RESEARCH.
