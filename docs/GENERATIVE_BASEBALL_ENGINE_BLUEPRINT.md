# Generative Baseball Engine Blueprint

Status: RESEARCH_ONLY architecture reference. This document does not declare the design implemented or production-ready.

## Target

The proposed engine targets a posterior predictive distribution for the complete game:

P(Y_game | D_pre)

rather than only a win classifier. Persistent talent, today's state, environment, decision policy and game randomness should remain distinguishable.

## Hard constraints

- No odds, market forecasts or bookmaker information.
- Point-in-time correctness is mandatory.
- retrieved_at, available_at and published_at remain distinct.
- Unknown availability is never converted into known availability.
- Random train/test splitting is prohibited.
- Candidate selection uses chronological OOS/WFO.
- Frozen holdout is never used for tuning.
- Research-only simulation cannot auto-promote production.
- Complexity must earn adoption through future-game evidence.

## Model layers

### Data and PIT

Prediction and evidence records should preserve game_id, game_time, prediction_time, cutoff, available_at, published_at, retrieved_at, revision_time, source, provenance status and data quality status. Stable IDs are required. Missing, unknown, unavailable, delayed, malformed and source-failed are distinct states.

### Latent talent

Represent hitters and pitchers as probability distributions over interpretable skills instead of a single rating. Use hierarchical shrinkage and partial pooling for small samples. Similar-player information is a conditional prior, not a copied rating.

### Current state

Separate persistent talent from current state. Candidate state variables include rest, workload, recent velocity/command changes, health uncertainty, lineup availability, bullpen availability, bench depletion and regime. A state-space/change-point layer can represent fatigue, injury return and mechanical changes.

### Matchup

Condition hitter/pitcher outcomes on pitch type, location, count, handedness, park and environment. Bound interaction complexity and use partial pooling.

### Lineup, defense and baserunning

Model actual participants, batting order, substitutions, defensive alignment, defender range/reaction/arm, baserunner speed and running-game interactions when reliable data exist. Uncertain participants remain scenario mixtures.

### Bullpen and manager policy

Bullpen state is dynamic: talent, availability, recent workload, fatigue and effectiveness conditional on game state. Manager behavior can be represented as:

P(Action | Game State)

Actions may include pitcher change, pinch hit, bunt, steal and intentional walk when sample support is sufficient.

### Game state and event generation

Represent:

S_t = (inning, half, outs, bases, count, score, batter, pitcher, pitch count, bullpen, bench)

and learn:

P(S_{t+1} | S_t, A_t)

Where pitch-level data support it, progressively model pitch type, velocity/movement, location, swing decision, contact, batted-ball characteristics, fielding outcome, baserunning and runs. A statistical generative model is preferred to a full physics simulator until evidence justifies the added cost.

### Environment and joint score distribution

Represent park, roof, temperature, wind, precipitation and umpire effects probabilistically when unresolved. Target the joint distribution:

P(R_home, R_away | D_pre)

rather than two independent expected-run values. A shared game-level latent factor can represent common run-environment shocks.

### Posterior predictive Monte Carlo

Use nested sampling only as justified by the model:

1. talent/state posterior;
2. environment and unresolved pregame scenarios;
3. game-level randomness.

Simulation count controls numerical Monte Carlo error, not model correctness.

### Scenario mixtures

For unresolved information:

P(Y) = sum_s P(Y | scenario_s) P(scenario_s)

For example, uncertain starter identity can be simulated as multiple explicit worlds and integrated with frozen scenario probabilities.

### Ensemble and calibration

Retain structurally different models when they provide independent signal: direct outcome classifier, hierarchical result model, state-transition/Game-Script model and generative simulation. Stacking/blending uses OOS predictions only. Calibration is fitted on chronological development data only.

### Uncertainty

Track parameter/talent uncertainty, model disagreement, information uncertainty, source/data uncertainty, lineup/starter uncertainty, environment uncertainty, regime uncertainty and simulation uncertainty separately. Confidence and predictability are different quantities. Candidate actions can include PREDICT, ACQUIRE_MORE, WAIT, RECOMPUTE, FALLBACK and ABSTAIN.

### Counterfactual and VOI

Counterfactual simulation can study starter swaps, closer swaps, lineup changes, bullpen changes and weather scenarios. It must not be described as causal effect without causal assumptions.

VOI can prioritize information acquisition:

VOI(X) = ExpectedUtility(with X) - ExpectedUtility(without X)

## Evaluation

Primary metric: LogLoss.

Secondary metrics: Accuracy, Brier, ECE/reliability and class-wise calibration.

Score metrics: run MAE, score-error distribution, exact-score Top-k, Low <=6, High >=7 and joint score-distribution quality.

Break down by league, season, phase, starter certainty, uncertainty tier, regime, data quality and forecast horizon.

Robustness requires feature deletion, source removal, time shift, distribution shift, new players/teams, high-uncertainty cases, rare/high-variance cases and PIT/leakage attacks.

## Integration order

1. Keep existing PIT/data contracts as the hard boundary.
2. Connect latent talent/state distributions to existing feature/result interfaces.
3. Add bullpen availability/fatigue and manager policy as research-only modules.
4. Strengthen Game-State/Game-Script transitions.
5. Add probabilistic score generation and joint score distributions.
6. Compare direct and generative outputs through the existing calibrated ensemble.
7. Add uncertainty decomposition and selective decisions.
8. Add counterfactual/VOI research.
9. Consider adoption only after chronological OOS, robustness, calibration, PIT and frozen-holdout gates pass.

## Existing repository foundations

The current main already contains PIT evidence/replay components, hierarchical NPB result modeling, Game-State research, Game-Script research and Monte Carlo, individually calibrated ensemble research, shared calibration evaluation, chronological OOS, governance and recovery workflows.

The correct strategy is therefore staged integration and empirical validation, not an uncontrolled rewrite into one giant model.
