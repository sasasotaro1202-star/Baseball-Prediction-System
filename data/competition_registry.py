"""Canonical competition registry for baseball research and production gating.

The registry is deliberately metadata-only: registering a competition never makes
it production eligible. Adapters must provide PIT-safe observations and the
competition must pass its own chronological OOS/holdout gates before promotion.
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
    notes: str = ""


COMPETITIONS: tuple[CompetitionSpec, ...] = (
    CompetitionSpec("NPB", "Nippon Professional Baseball", "professional", "top", "mixed", "Japan", "HOME_DRAW_AWAY", "PRODUCTION_ELIGIBLE", True),
    CompetitionSpec("MLB", "Major League Baseball", "professional", "top", "mixed", "United States/Canada", "HOME_AWAY", "RESEARCH_ONLY", True, notes="Historical starter announcement timestamp evidence remains a production gate."),
    CompetitionSpec("WBC", "World Baseball Classic", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("WBSC_PREMIER12", "WBSC Premier12", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("OLYMPICS_BASEBALL", "Olympic Baseball", "international_senior", "senior", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("ASIAN_GAMES_BASEBALL", "Asian Games Baseball", "international_senior", "senior", "mixed", "Asia", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("WBSC_U18", "WBSC U-18 Baseball World Cup", "international_youth", "U18", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("WBSC_U23", "WBSC U-23 Baseball World Cup", "international_youth", "U23", "mixed", "international", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("KOSHIEN_SENBATSU", "National High School Baseball Invitational (Senbatsu)", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("KOSHIEN_SUMMER", "National High School Baseball Championship (Summer Koshien)", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("KOSHIEN_QUALIFIERS", "Japanese High School Baseball Qualifiers", "high_school", "U18", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
    CompetitionSpec("JAPAN_UNIVERSITY_BASEBALL", "Japanese University Baseball", "collegiate", "university", "mixed", "Japan", "COMPETITION_DEFINED", "RESEARCH_ONLY", True),
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
