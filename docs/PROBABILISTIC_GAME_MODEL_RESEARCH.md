# Probabilistic Game Model Research Layer

## Status

RESEARCH_ONLY. This document describes an experimental integration of the supplied hierarchical generative design into the existing Baseball-Prediction-System architecture. It is not production evidence and does not change Champion/Production state.

## Integration principle

The repository already has research implementations for game-state transitions, Monte Carlo game simulation, correlated score distributions, hierarchical NPB result modelling, calibration, uncertainty, and model disagreement. The new layer therefore acts as an orchestration/contract layer rather than a second independent baseball simulator.

Target flow:

latent talent
→ current state
→ explicit pre-game scenario mixture
→ existing game-state/score simulator
→ posterior predictive distribution
→ uncertainty decomposition
→ model disagreement
→ calibration

## Latent talent and shrinkage

Observed short-run performance is not treated as identical to stable player ability. normal_normal_update() provides an explicit, auditable shrinkage mechanism where a prior is updated by an observed mean with observation-count-dependent precision.

This is a generic research primitive. It is not a claim that every baseball skill has a Gaussian likelihood; domain-specific player/pitch/batter models must be validated separately.

## Current state

apply_state_adjustment() keeps a transparent separation between a latent talent estimate and a current-state adjustment such as fatigue. Future integrations can bind this contract to PIT-safe rest, workload, availability, starter, lineup and bullpen state features.

## Scenario uncertainty

Unknown or changing pre-game information should not be silently collapsed to one deterministic value. Scenario allows mutually exclusive pre-game worlds such as starter X / starter Y, lineup configuration A / B, weather state A / B, and bullpen availability state A / B, with explicit normalized weights and provenance parameters.

## Posterior predictive simulation

posterior_predictive_run() accepts a simulate_fn that can reuse an existing game simulator and returns a coherent score distribution plus NPB HOME/DRAW/AWAY probabilities, MLB HOME/AWAY probabilities, LOW (total runs <= 6), HIGH (total runs >= 7), Top-4 exact-score candidates, Monte Carlo standard-error diagnostic, scenario-specific outcomes, scenario variance decomposition, and optional model disagreement.

Scenario allocation is deterministic and stratified while the inner game simulation remains stochastic. This improves reproducibility and prevents a low-weight scenario from disappearing entirely due to a zero multinomial draw.

## Uncertainty

The layer separates within-scenario outcome variance from between-scenario variance using the law of total variance. These labels are operational diagnostics, not claims that the decomposition perfectly identifies the real-world ontology of uncertainty.

Model disagreement is reported with probability variance and Jensen-Shannon divergence. This is intentionally separate from raw confidence.

## PIT and target safety

The research layer fails closed unless pit_status == PASS and available_at <= prediction_cutoff with timezone-aware timestamps. Unknown or unverifiable PIT is not converted to PASS.

NPB remains a three-class target (HOME/DRAW/AWAY). MLB remains binary (HOME/AWAY). A terminal MLB tie without an explicit extra-inning rule is rejected rather than silently mapped into the binary target.

No market/odds input is part of this research layer.

## Adoption gate

Implementation, unit tests, successful workflow execution, or simulation artifacts do not make this model performance-verified.

Any promotion candidate must still pass the repository normal sequence:

LOCAL PIT
→ chronological OOS/WFO
→ calibration
→ ablation
→ robustness
→ frozen holdout
→ release gate

Only measured evidence may change production status.

## Files

- research/probabilistic_game_model.py
- config/probabilistic_game_model.json
- tests/test_probabilistic_game_model.py

This design intentionally remains separate from the canonical PROJECT_SOURCE.md provenance lock.
