"""Canonical source registry and provenance metadata for NPB/MLB."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    league: str
    feature: str
    endpoint: str
    priority: int
    critical: bool
    availability_rule: str


SOURCES = (
    SourceSpec("npb_schedule", "NPB", "schedule_result_identity", "https://npb.jp/", 1, True, "source availability must be <= cutoff"),
    SourceSpec("npb_starters", "NPB", "starting_pitchers", "https://spaia.jp/baseball/npb/api/flash_atbat_history", 1, True, "announcement timestamp <= cutoff"),
    SourceSpec("npb_lineups", "NPB", "lineups", "https://spaia.jp/baseball/npb/api/starting_members_for_flash", 1, False, "announcement timestamp <= cutoff"),
    SourceSpec("mlb_statsapi", "MLB", "schedule_result_identity", "https://statsapi.mlb.com/api/v1", 1, True, "source availability must be <= cutoff"),
    SourceSpec("mlb_starters", "MLB", "starting_pitchers", "https://statsapi.mlb.com/api/v1", 1, True, "confirmed pre-first-pitch information only"),
    SourceSpec("statcast", "MLB", "pitch_level_and_pitcher_detail", "Baseball Savant / Statcast", 1, False, "publication/event information must be <= cutoff"),
    SourceSpec("fangraphs", "MLB", "advanced_batting_pitching_reference", "FanGraphs", 2, False, "historical metric must be cutoff-safe"),
    SourceSpec("weather", "NPB+MLB", "weather", "Open-Meteo", 2, False, "archived observation/forecast snapshot only"),
)


def registry() -> list[dict[str, Any]]:
    return [asdict(s) for s in SOURCES]


def resolve(league: str, feature: str) -> list[SourceSpec]:
    return [s for s in SOURCES if (s.league == league or s.league == "NPB+MLB") and s.feature == feature]
