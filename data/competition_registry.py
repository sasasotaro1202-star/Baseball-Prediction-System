"""Canonical competition registry for baseball research and production gating.

The registry is metadata-only: listing a competition never makes it production
eligible. Each competition family gets an explicit outcome contract and rules
profile so league play and tournament/knockout play are never mixed blindly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal["PRODUCTION_ELIGIBLE", "RESEARCH_ONLY", "UNAVAILABLE"]


@dataclass(frozen=True)
class CompetitionSpec:
    competition_id: str
    name: str
    competition_type: str
    level: str
    gender: str
    geography: str
    outcome_contract: str
    status: Status
    requires_starter_announcement: bool
    pit_required: bool = True
    phase_type: str = "league"
    rules_profile: str = "standard_baseball"
    notes: str = ""


COMPETITIONS: tuple[CompetitionSpec, ...] = (
    # League / regular-season competitions: keep these isolated and highest priority.
    # NPB remains fail-closed until the real-data candidate lifecycle has produced
    # an independent holdout-backed ADOPT decision. This prevents a metadata-only
    # registry entry from becoming a production promotion by accident.
    CompetitionSpec("NPB", "Nippon Professional Baseball", "professional", "top", "mixed", "Japan", "HOME_DRAW_AWAY", "RESEARCH_ONLY", True, phase_type="league", rules_profile="npb", notes="Production promotion requires candidate lock, independent holdout, PIT starter evidence, calibration, score/Low-High checks, and explicit ADOPT evidence."),
    CompetitionSpec("MLB", "Major League Baseball", "professional", "top", "mixed", "United States/Canada", "HOME_AWAY", "RESEARCH_ONLY", True, phase_type="league", rules_profile="mlb", notes="Historical starter announcement timestamp evidence remains a production gate."),

    # Senior national-team tournaments / non-league competition.
    CompetitionSpec("WBC", "World Baseball Classic", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="wbc"),
    CompetitionSpec("WBSC_PREMIER12", "WBSC Premier12", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="wbsc_senior"),
    CompetitionSpec("OLYMPICS_BASEBALL", "Olympic Baseball", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="olympic_baseball"),
    CompetitionSpec("ASIAN_GAMES_BASEBALL", "Asian Games Baseball", "international_senior", "senior", "mixed", "Asia", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="asian_games"),

    # Youth / age-group national-team competitions.
    CompetitionSpec("WBSC_U18", "WBSC U-18 Baseball World Cup", "international_youth", "U18", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="wbsc_age_group"),
    CompetitionSpec("WBSC_U23", "WBSC U-23 Baseball World Cup", "international_youth", "U23", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="wbsc_age_group"),

    # Japanese high-school baseball: qualifiers and national tournaments are separate phases.
    CompetitionSpec("KOSHIEN_SENBATSU", "National High School Baseball Invitational (Senbatsu)", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="japan_high_school"),
    CompetitionSpec("KOSHIEN_SUMMER", "National High School Baseball Championship (Summer Koshien)", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="tournament", rules_profile="japan_high_school"),
    CompetitionSpec("KOSHIEN_QUALIFIERS", "Japanese High School Baseball Qualifiers", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="qualifier", rules_profile="japan_high_school_qualifier"),

    # Collegiate / amateur competition.
    CompetitionSpec("JAPAN_UNIVERSITY_BASEBALL", "Japanese University Baseball", "collegiate", "university", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True, phase_type="league_and_tournament", rules_profile="japan_university"),
)


def get(competition_id: str) -> CompetitionSpec:
    for spec in COMPETITIONS:
        if spec.competition_id == competition_id:
            return spec
    raise KeyError(f"unknown competition_id: {competition_id}")


def registry() -> list[dict[str, object]]:
    return [asdict(spec) for spec in COMPETITIONS]


def production_eligible(competition_id: str) -> bool:
    return get(competition_id).status == "PRODUCTION_ELIGIBLE"
