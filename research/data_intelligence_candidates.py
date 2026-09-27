"""Research-only registry of baseball data-intelligence sources.

This is deliberately separate from the production/source registry.  It records
high-information sources that could make the system more Opta-like, while
preserving the free-first rule and preventing commercial sources from being
mistaken for available dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DataIntelligenceCandidate:
    candidate_id: str
    provider: str
    access_class: str
    coverage: str
    signal_classes: tuple[str, ...]
    prediction_use: tuple[str, ...]
    pit_notes: str
    free_usable_now: bool
    auto_acquire: bool = False


CANDIDATES: tuple[DataIntelligenceCandidate, ...] = (
    DataIntelligenceCandidate(
        "mlb_statcast",
        "Baseball Savant / Statcast",
        "FREE_PUBLIC",
        "MLB; selected MiLB; WBC",
        ("pitch", "batted_ball", "pitcher", "batter", "fielding", "running", "bat_tracking"),
        ("starter_form", "pitch_arsenal", "contact_quality", "defense", "speed", "regime"),
        "Event-time and availability boundaries must be preserved; not every derived field is prediction-time safe.",
        True,
    ),
    DataIntelligenceCandidate(
        "mlb_statsapi",
        "MLB Stats API",
        "FREE_PUBLIC",
        "MLB",
        ("schedule", "roster", "probable_pitcher", "game_state", "lineup", "player"),
        ("game_discovery", "starter_status", "roster_context", "live_revision"),
        "Use retrieved payloads only when their information was public by prediction cutoff.",
        True,
    ),
    DataIntelligenceCandidate(
        "mlb_gameday",
        "MLB Gameday / historical feed",
        "FREE_PUBLIC_ARCHIVAL",
        "MLB",
        ("at_bats", "pitch_fx", "rosters", "umpires", "game_state"),
        ("historical_pbp", "umpire_context", "state_transition"),
        "Legacy feed rights and historical publication timing must be retained as provenance.",
        True,
    ),
    DataIntelligenceCandidate(
        "sportsdataverse_mlb_models",
        "SportsDataverse MLB releases",
        "FREE_GITHUB",
        "MLB",
        ("xstats", "xera", "stuff_plus", "command_plus", "oaa", "framing", "game_state"),
        ("model_blend_prior", "pitcher_quality", "batter_quality", "defense", "uncertainty"),
        "Model output must be joined using an as-of boundary; training-era leakage must be audited.",
        True,
    ),
    DataIntelligenceCandidate(
        "sportsdataverse_mlb_raw",
        "SportsDataverse MLB raw capture",
        "FREE_GITHUB",
        "MLB",
        ("raw_statsapi", "raw_statcast", "game_manifest", "sha256_provenance"),
        ("reproducible_raw_layer", "PIT_reconstruction", "cross_source_replay"),
        "Per-game manifest and checksums provide strong provenance; historical capture availability still needs prediction-time reconstruction.",
        True,
    ),
    DataIntelligenceCandidate(
        "sportsdataverse_ncaa",
        "SportsDataverse baseballr-data / sportsdataverse-data",
        "FREE_GITHUB",
        "NCAA baseball",
        ("schedule", "pbp", "roster", "team_reference"),
        ("NCAA_scope_expansion", "historical_oos", "team_state"),
        "Historical release timing is not equivalent to prediction-time PIT; store release/publication evidence.",
        True,
    ),
    DataIntelligenceCandidate(
        "retrosheet",
        "Retrosheet",
        "FREE_PUBLIC",
        "MLB historical",
        ("parsed_pbp", "boxscore", "game_events"),
        ("long_history", "rare_case_research", "player_team_state"),
        "Primarily historical; never treat the archive itself as proof of live availability.",
        True,
    ),
    DataIntelligenceCandidate(
        "chadwick_register",
        "Chadwick Bureau Register",
        "FREE_GITHUB",
        "MLB / historical player identity",
        ("player_ids", "crosswalks"),
        ("identity_resolution", "cross_source_join"),
        "Identity mapping is not outcome information, but source/version provenance must be retained.",
        True,
    ),
    DataIntelligenceCandidate(
        "retrosplits",
        "Chadwick Bureau retrosplits",
        "FREE_GITHUB",
        "MLB historical",
        ("daybyday", "batting_splits", "pitching_splits"),
        ("state_features", "split_regime", "historical_priors"),
        "Historical aggregation is research-only until an as-of methodology is reconstructed.",
        True,
    ),
    DataIntelligenceCandidate(
        "baseballcv",
        "BaseballCV",
        "FREE_GITHUB",
        "MLB broadcast + amateur baseball research",
        ("computer_vision", "ball_tracking", "object_detection", "glove_tracking"),
        ("video_derived_features", "tracking_gap_fill", "visual_state"),
        "Dataset snapshots are not live evidence; applying models to current games requires a lawful/current video source.",
        True,
    ),
    DataIntelligenceCandidate(
        "openbiomechanics",
        "Driveline OpenBiomechanics",
        "FREE_RESEARCH_DATA",
        "Mostly collegiate-level research cohort",
        ("motion_capture", "kinematics", "kinetics", "force_plate", "swing_metrics"),
        ("player_prior", "injury_proxy_research", "mechanics_similarity"),
        "Use as a player/mechanics prior only; not game-time evidence. Commercial-use restrictions apply.",
        True,
    ),
    DataIntelligenceCandidate(
        "spaia_npb",
        "SPAIA NPB public endpoints",
        "FREE_PUBLIC_ENDPOINT_CANDIDATE",
        "NPB",
        ("schedule", "lineups", "pitch_by_pitch", "pbp", "player_stats", "pregame_odds"),
        ("NPB_game_discovery", "starter_context", "lineup_context", "market_context", "pbp"),
        "Historical PIT is not guaranteed by current endpoint access; archive available_at must be captured.",
        True,
    ),
    DataIntelligenceCandidate(
        "cpbl_public_2026",
        "CPBL official-site-derived public 2026 corpus",
        "FREE_GITHUB",
        "CPBL",
        ("schedule", "starting_pitcher", "starting_batting_order", "standings"),
        ("CPBL_game_discovery", "starter_context", "lineup_context"),
        "Third-party collection of official pages; original publication time must be reconstructed before OOS use.",
        True,
    ),
    DataIntelligenceCandidate(
        "kbo_public",
        "KBO public schedule/stats ecosystem",
        "FREE_PUBLIC",
        "KBO",
        ("schedule", "game_stats", "player_stats", "pbp_candidates"),
        ("KBO_game_discovery", "starter_context", "player_form"),
        "Use source-level retrieval/publication timestamps; tracking access is not assumed merely from existence of TrackMan.",
        True,
    ),
    DataIntelligenceCandidate(
        "omyu",
        "OmyuTech / 一球速報.com",
        "FREE_PUBLIC",
        "Japan high school, university, amateur, youth, independent",
        ("schedule", "boxscore", "pbp", "pitch_detail", "lineup"),
        ("scope_expansion", "lineup_context", "pitcher_batter_state"),
        "Organizer-dependent coverage and historical availability require source-specific PIT validation.",
        True,
    ),
    DataIntelligenceCandidate(
        "wbsc_official",
        "WBSC / MyWBSC / event reports",
        "FREE_PUBLIC",
        "International baseball and age-group events",
        ("schedule", "results", "rosters", "tournament_rules", "stats"),
        ("tournament_scope", "roster_prior", "rules_context"),
        "Coverage and granularity vary by event; historical PIT/PBP must be proven per event.",
        True,
    ),
    DataIntelligenceCandidate(
        "jaba_big6",
        "JABA + Big6 Scorebook",
        "FREE_PUBLIC",
        "Japan corporate/amateur/university",
        ("schedule", "results", "player_stats", "scorebook"),
        ("Japan_amateur_scope", "team_state", "player_form"),
        "Publication boundaries vary by organizer; do not infer unavailable timestamps.",
        True,
    ),
    DataIntelligenceCandidate(
        "fangraphs",
        "FanGraphs public leaderboards/projections",
        "FREE_PUBLIC_PAGE_WITH_RESTRICTIONS",
        "MLB",
        ("advanced_stats", "projections", "splits"),
        ("external_prior", "ensemble_benchmark", "player_strength"),
        "Some projection/history exports are member-only; only openly retrievable values may enter the free pipeline.",
        True,
    ),
    DataIntelligenceCandidate(
        "sportradar_global_baseball",
        "Sportradar Global Baseball v2",
        "COMMERCIAL_REFERENCE",
        "MLB, NPB, KBO, NCAA and more",
        ("schedule", "player", "team", "historical_results", "lineup", "win_probability"),
        ("schema_benchmark", "coverage_benchmark", "future_adapter_design"),
        "License/API key required; do not acquire automatically under free-only policy.",
        False,
    ),
    DataIntelligenceCandidate(
        "stats_perform_opta",
        "Stats Perform / Opta",
        "COMMERCIAL_REFERENCE",
        "Broad sports; MLB products",
        ("granular_events", "historical_data", "MLB_X_INFO", "prediction_products"),
        ("schema_benchmark", "feature_idea_generation", "prediction_benchmark"),
        "Commercial data feed; use only as architecture/feature benchmark unless a free legal source exists.",
        False,
    ),
    DataIntelligenceCandidate(
        "datastadium_dmp",
        "Data Stadium NPB DMP",
        "COMMERCIAL_REFERENCE",
        "NPB",
        ("tracking", "official_game_tracking"),
        ("NPB_tracking_feature_design", "schema_benchmark"),
        "2026 NPB Enterprise agreement enables tracking-data sales; license is not assumed.",
        False,
    ),
    DataIntelligenceCandidate(
        "trackman",
        "TrackMan Baseball Data API",
        "COMMERCIAL_REFERENCE",
        "Tracked stadium/portable sessions",
        ("pitch_tracking", "hit_tracking", "video", "positioning", "play_by_play_feed"),
        ("tracking_feature_design", "latency_benchmark"),
        "Data API and play-by-play access require customer credentials/access package.",
        False,
    ),
    DataIntelligenceCandidate(
        "rapsodo",
        "Rapsodo / Rapsodo Cloud",
        "COMMERCIAL_REFERENCE",
        "Player/session biomechanics and ball-flight measurement",
        ("pitch_metrics", "hit_metrics", "video", "player_measurement"),
        ("mechanics_prior", "player_quality", "player_development_signal"),
        "Hardware/cloud usage is subscription-linked; not a free production dependency.",
        False,
    ),
    DataIntelligenceCandidate(
        "sports_info_solutions",
        "Sports Info Solutions",
        "COMMERCIAL_REFERENCE",
        "Multiple baseball levels",
        ("advanced_stats", "historical_data", "data_feeds"),
        ("schema_benchmark", "coverage_benchmark", "feature_ideas"),
        "Commercial feed/client access; no automatic acquisition.",
        False,
    ),
)


def candidate_rows() -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in CANDIDATES]


def free_research_candidates() -> tuple[DataIntelligenceCandidate, ...]:
    return tuple(candidate for candidate in CANDIDATES if candidate.free_usable_now)


def commercial_reference_candidates() -> tuple[DataIntelligenceCandidate, ...]:
    return tuple(candidate for candidate in CANDIDATES if not candidate.free_usable_now)


def main() -> int:
    import json
    print(json.dumps({
        "candidate_count": len(CANDIDATES),
        "free_research_count": len(free_research_candidates()),
        "commercial_reference_count": len(commercial_reference_candidates()),
        "candidates": candidate_rows(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
