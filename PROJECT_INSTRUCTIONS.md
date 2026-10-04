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
Prediction-time competition/phase taxonomy captured by the production snapshot must survive postgame reconciliation into the experience ledger. Reconciliation may enrich missing taxonomy only from the immutable prediction snapshot or independently verified prediction-time evidence; it must not infer phase from postgame knowledge or silently remap UNKNOWN.

## Completion
Green Actions or generated artifacts are execution evidence, not automatic performance verification. Completion requires tests, PIT/leakage, identity/scope audit, chronological OOS, calibration, ablation, robustness, frozen holdout, artifact integrity, reproducibility, recovery, release gate, monitoring and rollback.

## Loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY → NEXT RESEARCH.

## Mandatory GitHub source for NPB prediction requests
Whenever a user asks for an NPB prediction, today's NPB forecast, a daily NPB forecast, or otherwise requests an NPB game prediction, the answer MUST use the prediction produced by the current GitHub repository `sasasotaro1202-star/Baseball-Prediction-System` as the primary prediction source. Before presenting any NPB prediction, re-check the repository's current default branch/HEAD and the applicable production prediction path/artifact for the requested JST date. The GitHub-derived probability, selected outcome, Top-4 exact scores, and Low/High output are authoritative for that request unless the repository marks the output BLOCKED, UNKNOWN, UNVERIFIABLE, FALLBACK, ABSTAIN, or otherwise ineligible.

External web/current-game information may be used only as a separate verification/context layer and must never be silently substituted for the GitHub prediction. It may not override the GitHub model output unless the response explicitly states that the GitHub output is unavailable/invalid and the result is therefore not a GitHub-based prediction. If a valid current GitHub prediction artifact/output cannot be located or generated, do not label an independently generated forecast as the GitHub prediction; report the GitHub prediction status as UNAVAILABLE/UNVERIFIABLE and preserve the distinction between repository evidence and external inference.

Every such response should retain, when available, the GitHub commit SHA, model version, feature/calibration version, prediction timestamp/cutoff, data snapshot or artifact identifier, and production/fallback status so the user can distinguish current GitHub evidence from live contextual information.

## Candidate OOS stale-run recovery
Candidate OOS keeps `cancel-in-progress: false` so a live chronological validation is never interrupted. The watchdog may recover only a same-main-SHA `in_progress` Candidate OOS run older than 360 minutes, which is deliberately beyond the 260-minute workflow timeout. Cancellation failure is fail-closed and blocks duplicate dispatch. A verified stale-run recovery triggers an immediate current-main redispatch. This operational recovery rule does not relax PIT, chronological OOS, calibration, holdout, adoption, or production gates.

## Candidate OOS evidence freshness
Any change to a module that can alter Candidate OOS selection, calibration, scoring, routing, adoption gating, or candidate identity must retrigger the Candidate OOS workflow before its evidence can be treated as current. Trigger coverage is part of the evidence-integrity contract; green CI from an older snapshot is not evidence for the changed implementation.

### Long-running OOS versus append-only experience commits
An already-running Candidate OOS replay may tolerate a main-branch commit only when the complete diff is limited to non-runtime continuity files: `data/experience/**`, `tests/**`, or `.github/workflows/baseball_candidate_oos_watchdog.yml`. Experience files are append-only historical postgame reconciliation outputs; tests and the candidate-OOS watchdog are verification/control-plane files and are not runtime inputs to candidate selection or scoring. Any code, candidate workflow, configuration, PIT source, evaluation, or other data change remains evidence-affecting and must fail closed. This distinction prevents operational/test maintenance from invalidating otherwise valid long-running OOS evidence without weakening the code/config snapshot gate.
