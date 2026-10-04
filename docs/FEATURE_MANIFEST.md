# Baseball Prediction System — Feature Manifest

## Version
feature-contract-v1

## Meaning of used
This manifest distinguishes ACTIVE, CONDITIONAL, OBSERVATION_ONLY, and RESEARCH_CANDIDATE features.

ACTIVE = included in the model feature matrix under the current production path.
CONDITIONAL = included only when explicit PIT-safe context configuration and timestamps permit it.
OBSERVATION_ONLY = collected/stored for analysis or evidence but not fed into production probability.
RESEARCH_CANDIDATE = available to research workflows but not production-authorized.

## Current NPB production feature matrix
The current baseball_backtest.py match_features path produces 482 scalar feature columns for NPB before conditional lineup/weather context features.

### 1. Team recent-form — ACTIVE
For each team, windows 3, 5, 10, 20, 30, 45, 60:
- pts_w, gf_w, ga_w, gd_w, win_w, draw_w
Home/away prefixes are h_ and a_; d_ columns are home-minus-away differences.

### 2. Shrunk team-form — ACTIVE
For windows 3, 5, 10, 20:
- win_shrunk_w
- gd_shrunk_w
- NPB: draw_shrunk_w

### 3. Home/away venue history — ACTIVE
- venue_n
- venue_pts
- venue_gf
- venue_ga

### 4. Team strength and schedule — ACTIVE
- elo
- rest_days
- matches
- bp3
- bp7

### 5. Team batting — ACTIVE
For windows 3, 5, 10, 20, 30:
- bat_ab_w
- bat_avg_w
- bat_hr_w
- bat_bb_w
- bat_so_w
- bat_bb_rate_w
- bat_so_rate_w
- bat_xbh_w
- bat_hr_rate_w
- bat_iso_proxy_w
- bat_extra_base_rate_w

### 6. Bullpen / relief — ACTIVE where source-supported
- bp_er_10
- bp_runs_10
- bp_app_10
- bp_ip_10
- bp_era_10
- bp_whip_10
- bp_k9_10
- bp_bb9_10
- bp_hr9_10
- bp_actual_coverage_10

The current NPB production path always has lagged bullpen workload proxies bp3/bp7. Explicit relief-quality metrics require source fields. Missing explicit relief data must be treated as a coverage issue, not as a claim that the true value is zero.

### 7. Volatility / trend — ACTIVE
For gf, ga, hr, so, bb over recent 20-game history:
- metric_sd_20
- metric_slope_20

### 8. Starting pitcher — ACTIVE when starter PIT gate passes
For both starters:
- era, whip, k9, bb9, hr9, fip, starts
- recent_era, recent_k9, recent_k_rate, recent_bb_rate
- recent_pitches, recent_ip
- era_slope, k9_slope, fip_slope
- recent_era_sd, recent_k9_sd, recent_pitches_sd
Prefixes: hs_ = home starter; as_ = away starter.

### 9. Matchup / derived — ACTIVE
- home_adv
- expected_env
- matchup_home_bat_vs_away_fip
- matchup_away_bat_vs_home_fip
- bullpen_fatigue_diff
- bullpen_quality_era_diff
- bullpen_whip_diff
- bullpen_k9_diff
- bullpen_bb9_diff
- bullpen_hr9_diff
- bullpen_actual_coverage_diff
- starter_x_quality_proxy
- starter_recency_gap
- starter_experience_gap
- starter_kbb_gap
- starter_hr_gap
- starter_recent_form_gap
- offense_power_gap_10
- offense_walk_gap_10
- offense_contact_gap_10
- run_volatility_gap_20
- run_trend_gap_20
- elo_x_starter_quality_gap
- elo_x_starter_reliability_gap
- form_x_rest_gap
- bullpen_fatigue_x_rest_gap
- offense_power_x_starter_quality
- environment_x_volatility_gap
- starter_quality_reliability_gap
- starter_known

### 10. Lineup context — CONDITIONAL, not current default production
Only when PIT_SAFE_CONTEXT_DATA=1 and lineup_announced_at and weather_available_at are proven at or before the prediction cutoff.
For each side:
- lineup_avg, lineup_obp, lineup_slg, lineup_iso
- lineup_bb_rate, lineup_so_rate, lineup_sb_rate
- lineup_bunt_rate, lineup_power_rate, lineup_contact_rate
- lineup_gidp_rate, lineup_groundout_rate, lineup_flyout_rate, lineup_lineout_rate
- lineup_errors_rate
- lineup_recent_iso, lineup_recent_so_rate, lineup_recent_bb_rate, lineup_recent_sb_rate
- lineup_n, lineup_known_n, lineup_history_coverage
- lineup_power_sum, lineup_speed_sum, lineup_bunt_sum, lineup_contact_sum, lineup_gdp_sum
Fully populated, this adds 81 lineup-related home/away/difference columns.

### 11. Weather context — CONDITIONAL, not current default production
- weather_temp_c
- weather_humidity_pct
- weather_wind_kmh
- weather_precip_mm
- weather_run_signal
The collector can store weather without feeding it into production probabilities.

## Base feature-column counts
Without conditional lineup/weather context:
- NPB: 482
- MLB: 470
NPB is larger because draw_shrunk_w exists.
These are contract-level expected counts; production output should record the actual feature count and schema hash.

## Classification model consumers
- Logistic Regression
- HistGradientBoosting
- RandomForest
- ExtraTrees
- KNNAnalog
- NPB HierarchicalDrawResult
- LightGBM when installed
- XGBoost when installed
- CatBoost when installed
The chronological ensemble may select only a subset.

## Score model consumers
- Poisson
- Tweedie
- HistGradientBoostingRegressor with Poisson loss
- RandomForestRegressor
- ExtraTreesRegressor

## Observation-only context by default
- detailed official NPB player profiles
- team-wide player context
- date-scoped roster context
- standings snapshots
- current weather snapshots unless PIT-safe context is enabled
- player identity/enrichment evidence
- source-health diagnostics
These do not automatically change production probabilities.

## Integrity
- Feature assembly order is deterministic.
- Every d_ feature must equal corresponding h_ minus a_.
- Same-timestamp game state is frozen before results are advanced.
- Target-game/postgame information cannot enter pregame features.
- Missing, unavailable, delayed, unknown and source-failed are distinct states.
- Adding a feature is not a success claim without chronological OOS, robustness and holdout evidence.

## Feature-set variants are first-class

The system must not treat the 482/470 base column counts as a universal fixed feature set. They are the current `match_features()` contract for the examined base path, and can change when the runtime, league, context gate, data availability, or validated experiment changes.

A feature set is identified by:
- league
- competition
- target contract/version
- runtime/model lane
- PIT-safe context mode
- feature-family allow-list
- feature version
- ordered feature-schema hash
- data/source snapshot identifiers

Examples of legitimate variants:
- BASELINE_TEAM_STATE
- TEAM_FORM_PLUS_ELO
- TEAM_PLUS_STARTER
- TEAM_PLUS_BULLPEN
- TEAM_PLUS_LINEUP_PIT_SAFE
- TEAM_PLUS_WEATHER_PIT_SAFE
- FULL_VALIDATED_ENSEMBLE
- SCORE_MODEL_FEATURE_SET
- RESEARCH_STATCAST_SET

These names are illustrative registry labels, not proof that every variant is currently implemented or production-eligible.

Research may compare many feature sets, but each candidate must be separately labeled and evaluated. Adding more features is not inherently better. A smaller feature set with stronger temporal stability, lower missingness and better OOS calibration may be preferred.

For every generated prediction, the artifact should record at minimum:
- feature_set_id
- feature_manifest_version
- feature_count
- feature_schema_hash
- feature_context_mode
- feature_data_quality/status
- source/data snapshot identifiers

The actual runtime artifact, not this document, is the authoritative record of the exact feature columns consumed for that prediction.

