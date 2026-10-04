# PROJECT_INSTRUCTIONS — Baseball-Prediction-System

## Mission
Maximize future-game generalization, case-level correctness, calibration, predictability awareness, uncertainty quality, robustness, PIT integrity and operational reliability for NPB/MLB and other validated baseball competitions.

## Every run
Re-check latest GitHub HEAD/default branch, code/config, tests, workflows, Actions, artifacts, registries, production/champion/challenger state, OOS/holdout evidence, failures and backlog. Prefer REUSE → REPAIR → INTEGRATE → TEST → VERIFY. Never rewrite historical evidence.

## PIT / time
Separate game_time, prediction_time, cutoff, source availability/publication, retrieval and revision. Starter, lineup, bullpen, weather, market and player-availability information is time-varying. Later confirmations or corrections cannot flow backward. Postgame/finalized same-game statistics are prohibited from pregame features.

## Target isolation
NPB uses HOME/DRAW/AWAY; MLB uses HOME/AWAY. Other competitions use their registered competition-specific contract. Score distributions, exact-score candidates and LOW/HIGH are separate target contracts. Do not convert incompatible targets silently.

## Evaluation
Chronological WFO/OOS only; random temporal splits are prohibited. Candidate selection and final OOS are separate. Frozen holdout is final evidence only and must not be tuned.

## Models / routing
Maintain naive/class-frequency/Elo/logistic baselines. Complex models, starter experts, Statcast experts and specialist routing require incremental OOS value, adequate sample/folds, calibration evidence and robustness. Sparse scopes fallback to broader validated models. Research-only models never become production merely because they produce output.

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

## Global GitHub-backed baseball prediction request contract
Whenever a user asks for a baseball game prediction, including phrases such as "試合予想して", "今日の試合を予想して", "NPB予想", "MLB予想", tournament prediction, or a named baseball matchup, the primary prediction source MUST be the current GitHub repository `sasasotaro1202-star/Baseball-Prediction-System`. This rule applies to NPB, MLB, international senior/youth tournaments, Japanese high-school/college competitions, and any other competition explicitly registered by the repository.

Before presenting any game prediction:
1. Re-check the current default branch/HEAD.
2. Identify the requested competition and target JST date/game.
3. Check whether a current valid GitHub prediction output/artifact already exists for that exact request.
4. If no valid prediction exists, do NOT stop merely because the artifact is missing. Use the repository's prediction-request generation path: create/update `prediction_requests/inbox/active.json` with the request, which triggers the GitHub Actions prediction-request workflow, or use the repository's equivalent workflow-dispatch path when a direct dispatch capability exists.
5. The GitHub request router must choose the strongest eligible lane in this order: CURRENT_PRODUCTION_RUNTIME → VALIDATED_RESEARCH_SHADOW → explicitly registered competition-specific research runtime → BLOCKED/UNAVAILABLE. It must never silently treat a research result as production.
6. Verify the generated request-result JSON and its status before treating it as usable evidence.
7. If generation is blocked, incomplete, PIT-unverifiable, starter/personnel-ineligible, or the expected GitHub result cannot be verified, report UNAVAILABLE/UNVERIFIABLE instead of substituting an independently generated forecast.

External web/current-game information may be used only as a separate context/verification layer. It may supplement the GitHub prediction with current schedule, confirmed starters, weather, roster and source-health facts, but it must not silently replace or override the GitHub model output.

Every response should preserve, when available: GitHub commit SHA, request id/fingerprint, model/runtime lane and version, feature/calibration version, prediction timestamp/cutoff, data/source snapshot identifiers, PIT status, generation status and production/research/fallback state. Evidence from GitHub and external context must remain explicitly separated.

A successful GitHub Action, existing workflow, code presence, or generated file alone never proves model performance. Performance claims remain governed by OOS/WFO, calibration, robustness, holdout and adoption gates.

## Prediction generation safety
The request-generation workflow is a controlled execution mechanism, not a promotion mechanism. It may generate a user-requested prediction with a validated research lane when a production runtime is unavailable, but the output must be labeled RESEARCH_SHADOW or other exact research status. It must never mutate Champion/Production state, alter historical evidence, bypass PIT, or convert UNKNOWN into PASS.

The request queue is append-oriented and idempotent by request fingerprint. Repeating an identical request should reuse a previously generated verified result rather than recompute it unnecessarily. Concurrent requests must be serialized or otherwise written deterministically so one request cannot overwrite another.

## Candidate OOS current-main automation
GitHub Actions includes an existing Candidate OOS watchdog (`baseball_candidate_oos_watchdog.yml`) on a 15-minute schedule. It reconciles queued/pending/in-progress Candidate OOS runs against the current main SHA and automatically dispatches a fresh current-main replay when evidence is stale or a bounded recovery condition is met.

Continuity-only changes such as append-only experience reconciliation, tests, documentation and explicitly validated control-plane workflow edits do not justify an expensive Candidate OOS replay. Changes to the Candidate OOS workflow itself, research/runtime code, configuration, dependencies, PIT data/source contracts or evaluation logic remain evidence-affecting and require a fresh current-main replay.

## Candidate OOS stale-run recovery
Candidate OOS keeps `cancel-in-progress: false` so a live chronological validation is never interrupted. The watchdog may recover only a same-main-SHA `in_progress` Candidate OOS run older than 360 minutes, which is deliberately beyond the 260-minute workflow timeout. Cancellation failure is fail-closed and blocks duplicate dispatch. A verified stale-run recovery triggers an immediate current-main redispatch. This operational recovery rule does not relax PIT, chronological OOS, calibration, holdout, adoption, or production gates.

## Candidate OOS evidence freshness
Any change to a module that can alter Candidate OOS selection, calibration, scoring, routing, adoption gating, or candidate identity must retrigger the Candidate OOS workflow before its evidence can be treated as current. Trigger coverage is part of the evidence-integrity contract; green CI from an older snapshot is not evidence for the changed implementation.

### Long-running OOS versus append-only experience commits
An already-running Candidate OOS replay may tolerate a main-branch commit only when the complete diff is limited to non-runtime continuity files: `data/experience/**`, `tests/**`, or `.github/workflows/baseball_candidate_oos_watchdog.yml`. Experience files are append-only historical postgame reconciliation outputs; tests and the candidate-OOS watchdog are verification/control-plane files and are not runtime inputs to candidate selection or scoring. Any code, candidate workflow, configuration, PIT source, evaluation, or other data change remains evidence-affecting and must fail closed. This distinction prevents operational/test maintenance from invalidating otherwise valid long-running OOS evidence without weakening the code/config snapshot gate.

## Feature contract and runtime evidence
The canonical feature manifest is `docs/FEATURE_MANIFEST.md` and the machine-readable policy is `config/feature_policy.json`. Feature status must be distinguished as ACTIVE, CONDITIONAL, OBSERVATION_ONLY, or RESEARCH_CANDIDATE. Do not infer production usage from a feature name appearing in source code.

The current default model matrix is expected to contain 482 NPB features and 470 MLB features before conditional lineup/weather context. Production prediction output should record actual feature count, feature-schema hash, manifest version and context mode. Feature assembly order must be deterministic.

Lineup/weather context is not production-active by default. It requires explicit PIT-safe configuration and cutoff-valid availability evidence. Observation-only player/roster/standings/context snapshots must not silently alter production probabilities.

Any feature change capable of changing prediction values is evidence-affecting and requires the normal TEST → PIT → chronological OOS/WFO → calibration → ablation → robustness → frozen holdout → release gate before adoption.

## Multiple feature-set variants
Feature sets are first-class versioned objects. Never assume a single universal feature list or that a larger feature set is better.

A prediction may use a variant selected from validated families such as BASELINE_TEAM_STATE, TEAM_PLUS_STARTER, TEAM_PLUS_BULLPEN, TEAM_PLUS_LINEUP_PIT_SAFE, TEAM_PLUS_WEATHER_PIT_SAFE, FULL_VALIDATED_ENSEMBLE, SCORE_MODEL_FEATURE_SET, or RESEARCH_STATCAST_SET. These labels are registry concepts; actual implementation and production eligibility must be verified from current code/config.

The exact features consumed by a prediction are identified by feature_set_id + ordered feature_schema_hash + feature_manifest_version + context mode. Production output must expose these fields. A feature-family snapshot, research candidate list, or collector output is not evidence of production consumption.

Feature-set selection must be chronological-OOS/PIT/calibration/robustness/holdout driven and may prefer a smaller, more stable set over a larger one.

## Ultimate research pattern laboratory
Use `research/ultimate_pattern_lab.py` for broad research-only pattern exploration. Stage A enumerates every subset of the eight optional feature families (256 patterns) on an early chronological OOS band. Stage B retests the top eight across ALL/SHORT/LONG core horizons, five recency half-lives and three model pools on a disjoint later OOS band. Stage C uses full ensemble/routing/calibration on a third disjoint band. Only the Development-selected Stage-C winner may be scored on the newest locked holdout, exactly once. No artifact from this lab can auto-promote Champion/Production.
## Extreme representation research
The research system also includes `research/extreme_representation_lab.py`. It explores fixed mathematical representations of the existing PIT-safe feature matrix (level, home/away-only, gap-only, absolute-gap, squared-gap, signed-log gap, level+gap transforms, and gated log-ratios). Representation selection is research-only and must use chronological OOS; the newest holdout remains locked and winner-only.

## Ultimate breadth expansion
The extreme representation laboratory may cross selected PIT-safe feature families and representations with deterministic model profiles. Such exploration remains RESEARCH_ONLY and cannot auto-promote Production.
## MLB starter PIT provenance
Starter evidence keeps announcement, publication, availability, retrieval, observation and revision timestamps distinct.
A probable-pitcher observation from MLB's current official page is not treated as an announcement timestamp. Production remains fail-closed until explicit announcement/availability provenance is present and passes the timestamp ordering gates.

## Score-distribution research
Use `research/score_distribution_pattern_lab.py` for research-only exploration of score-model composition, shared scoring correlation, and mean shrinkage. The canonical targets remain independent: Score Top-4 exact-score, Low<=6, High>=7. Candidate selection is chronological Development OOS only and the newest 20% remains winner-only frozen holdout. No score-pattern result may auto-promote Production.
