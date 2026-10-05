# Generative Baseball Engine Blueprint

## Status

- role: RESEARCH_ONLY architecture reference
- production eligibility: false
- market / odds / bookmaker inputs: forbidden
- PIT: every training/evaluation path must prove availability at prediction cutoff; otherwise UNKNOWN/UNVERIFIABLE and blocked
- validation: chronological OOS/WFO + robustness + frozen holdout
- adoption: candidate only; no automatic promotion

## Purpose

This document translates the proposed Bayesian hierarchical state-space generative baseball system into a staged architecture that can be integrated with the repository without replacing validated production components prematurely.

The target is not a single win classifier. The target is a posterior predictive distribution for the game:

P(Y_game | D_pre)

where latent player/team capability, current state, environment, decisions and game events are represented separately.

## Layered model

### 1. Point-in-time data and identity

Every observation should retain at least:

- game_id / stable entity IDs
- game_time
- prediction_time
- cutoff
- available_at
- published_at
- retrieved_at
- revision_time
- source
- provenance status
- data quality status

retrieved_at, available_at, and published_at are different fields.

Unknown availability is never converted into confirmed availability.

### 2. Latent talent

Represent player and team capability as distributions, not single ratings.

Hitters can include contact, zone contact, chase, power, exit velocity, barrel tendency, launch-angle profile, speed, platoon sensitivity, pitch-type sensitivity and location interactions.

Pitchers can include velocity, movement, spin, command/control, whiff, ground-ball tendency, contact-quality suppression, pitch mix, pitch-type quality and platoon effects.

Use hierarchical shrinkage for small samples and partial pooling across league, age, role, position and player type.

Do not copy a comparable player directly; infer a conditional prior distribution.

### 3. Current-state model

Separate persistent talent from today's state.

Examples:

- rest
- workload
- recent velocity/command changes
- health uncertainty
- lineup availability
- bullpen availability
- bench depletion
- season/regime state

Conceptually:

Observed = Talent + State + Noise

A latent state-space / change-point layer can detect persistent transitions such as fatigue, injury return or mechanical changes.

### 4. Matchup model

Condition hitter and pitcher distributions jointly on:

- pitch type
- location
- handedness
- count
- batter/pitcher skill
- park and environment

Share statistical strength across pitch types while keeping interaction terms bounded to avoid high-dimensional overfit.

### 5. Lineup / defense / baserunning

Model actual participants, not only team-average strength.

Include, when data support it:

- batting order
- substitutions
- defensive alignment
- defender range/reaction/arm
- baserunner speed
- catcher/pitcher running-game effects

Uncertain lineup identity remains a scenario mixture instead of an artificial certainty.

### 6. Bullpen and manager policy

Bullpen state is dynamic.

Represent each reliever with:

- latent talent
- availability
- recent workload
- current fatigue
- expected effectiveness conditional on state

Then model manager policy as P(Action | Game State).

Actions can include pitcher change, pinch hit, bunt, steal and intentional walk where sample support is sufficient.

A static bullpen ERA or average reliever rating is not a substitute for usage policy.

### 7. Game-state transition model

Represent:

S_t = (inning, half, outs, bases, count, score, batter, pitcher, pitch count, bullpen, bench)

and learn:

P(S_{t+1} | S_t, A_t)

The existing Game-State / Game-Script research is the foundation for this layer. Future work should preserve its PIT gate and fail-closed behavior.

### 8. Pitch / plate-appearance generation

Where sufficient pitch-level data exist, progressively refine the transition:

Pitch type -> velocity/movement -> location -> swing decision -> contact -> batted-ball characteristics -> fielding outcome -> baserunning -> runs.

Do not require a full physics simulator for the first implementation. A statistical generative approximation is acceptable when it improves OOS performance at bounded cost.

### 9. Environment

Represent uncertain environment probabilistically:

- park
- roof
- temperature
- wind
- precipitation
- umpire tendencies

Uncertain environment becomes a scenario distribution rather than an arbitrary point estimate.

### 10. Game-level shared latent factors

A shared game latent variable can induce realistic dependence between the two teams' run distributions.

The important target is the joint distribution:

P(R_home, R_away | D_pre)

not only two independent expected-run estimates.

### 11. Posterior predictive Monte Carlo

Use nested sampling only where it is useful:

1. sample talent/state posterior;
2. sample environment and unresolved pregame scenarios;
3. simulate the game path.

The number of simulations controls Monte Carlo numerical error, not model correctness.

Therefore Monte Carlo precision must never be reported as model accuracy.

### 12. Scenario mixtures

For unresolved pregame information:

P(Y) = sum_s P(Y | scenario_s) P(scenario_s)

Evaluate each defensible scenario and integrate it using frozen scenario probabilities.

### 13. Ensemble and calibration

Keep structurally different models when they add information:

- direct outcome classifier
- hierarchical result model
- state-transition / Game-Script model
- generative simulation model

Use out-of-sample predictions for stacking or blending.

Calibration parameters are learned only on chronological development data, never on the frozen holdout.

The repository's existing individually calibrated ensemble is the natural entry point.

### 14. Uncertainty decomposition

Do not collapse every source of uncertainty into one confidence number.

Track at least:

- parameter/talent uncertainty
- model disagreement
- information uncertainty
- source/data uncertainty
- lineup/starter uncertainty
- environment uncertainty
- regime uncertainty
- simulation uncertainty

Confidence and predictability are distinct.

Possible decisions:

PREDICT / ACQUIRE_MORE / WAIT / RECOMPUTE / FALLBACK / ABSTAIN

### 15. Counterfactuals and Value of Information

Counterfactual analysis is a simulation tool unless causal assumptions are separately established.

Useful research questions include starter swaps, closer swaps, batter removal, lineup-order changes, bullpen-availability changes and weather scenarios.

Value of Information can prioritize acquisition:

VOI(X) = ExpectedUtility(with X) - ExpectedUtility(without X)

Prioritize information that materially changes calibrated future performance.

## Evaluation contract

Every candidate must be compared on the same chronological splits.

Primary:

- LogLoss

Secondary:

- Accuracy
- Brier
- ECE / reliability
- class-wise calibration

Score distribution:

- run MAE
- run-error distribution
- exact-score Top-k
- Low (<=6) / High (>=7) distribution metrics
- joint home/away score-distribution quality

Required breakdowns:

- NPB / MLB
- season
- phase
- starter certainty
- uncertainty tier
- regime
- data quality
- forecast horizon

Required robustness checks:

- feature deletion
- source removal
- time shift
- distribution shift
- new player/team
- high uncertainty
- rare/high-variance cases
- PIT/leakage attack

## Integration order for this repository

1. Stabilize PIT/data contracts.
2. Connect latent talent/state distributions to existing feature and result-model interfaces.
3. Extend bullpen availability/fatigue and manager policy as research-only modules.
4. Strengthen Game-State/Game-Script transitions.
5. Add probabilistic score generation and joint score distributions.
6. Compare direct vs generative predictions through the existing calibrated ensemble.
7. Add uncertainty decomposition and selective decisions.
8. Add counterfactual/VOI experiments.
9. Adopt only components that pass chronological OOS, robustness, calibration, PIT and frozen-holdout gates.

## Non-goals

- No odds, market predictions or bookmaker information.
- No silent PIT backdating.
- No random train/test split.
- No production promotion from research-only simulation.
- No requirement to implement every pitch-physics detail before OOS evidence supports it.
- No complexity increase without measurable future-generalization value.

## Current repository mapping

Already present as research foundations:

- core/pit.py and core/pit_evidence.py
- core/pit_replay.py
- research/hierarchical_result_model.py
- research/game_state_engine.py
- research/game_script_* modules
- research/individually_calibrated_ensemble.py
- evaluation/calibration.py
- chronological candidate OOS / governance / recovery workflows

This blueprint is a research direction and interface-level target. It does not declare the full generative system implemented or production-ready.
