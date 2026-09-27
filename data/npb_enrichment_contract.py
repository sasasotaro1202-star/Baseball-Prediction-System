"""Shared contract for NPB multi-source enrichment fields.

These fields describe completed-game observations. They may be carried forward into
later chronological state, but target-game use still requires an independent PIT
check (especially lineups/weather/starter identity).
"""
from __future__ import annotations

ENRICHED_GAME_FIELDS = (
    "league", "venue", "start_time",
    "home_starter_line_ok", "away_starter_line_ok",
    "home_starter_era", "away_starter_era", "home_starter_whip", "away_starter_whip",
    "home_starter_k9", "away_starter_k9", "home_starter_bb9", "away_starter_bb9",
    "home_starter_hr9", "away_starter_hr9", "home_starter_fip", "away_starter_fip",
    "home_starter_ip", "away_starter_ip", "home_starter_er", "away_starter_er",
    "home_starter_h", "away_starter_h", "home_starter_hr", "away_starter_hr",
    "home_starter_bb", "away_starter_bb", "home_starter_so", "away_starter_so",
    "home_starter_pitches", "away_starter_pitches",
    "home_starter_k_rate", "away_starter_k_rate", "home_starter_bb_rate", "away_starter_bb_rate",
    "home_bat_pa", "away_bat_pa", "home_bat_ab", "away_bat_ab",
    "home_bat_h", "away_bat_h", "home_bat_hr", "away_bat_hr",
    "home_bat_bb", "away_bat_bb", "home_bat_so", "away_bat_so",
    "home_bat_2b", "away_bat_2b", "home_bat_3b", "away_bat_3b",
    "home_bat_sb", "away_bat_sb", "home_bat_cs", "away_bat_cs",
    "home_bullpen_er", "away_bullpen_er", "home_bullpen_ip", "away_bullpen_ip",
    "home_bullpen_h", "away_bullpen_h", "home_bullpen_bb", "away_bullpen_bb",
    "home_bullpen_so", "away_bullpen_so", "home_bullpen_hr", "away_bullpen_hr",
    "home_lineup_json", "away_lineup_json", "player_rows_count",
    "weather_temp_c", "weather_humidity_pct", "weather_precip_mm", "weather_wind_kmh",
)
