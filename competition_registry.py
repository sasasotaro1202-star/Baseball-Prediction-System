"""Stable competition registry and production eligibility metadata.

The registry is deliberately conservative: a competition can be known to the
system without being production-eligible. Unsupported competitions must stay
research-only until PIT coverage, result integrity, rules, and pregame
personnel requirements are independently verified.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

CompetitionStatus = Literal["PRODUCTION", "RESEARCH_ONLY", "UNAVAILABLE"]


@dataclass(frozen=True)
class CompetitionSpec:
    competition_id: str
    name: str
    sport: str
    level: str
    region: str
    ruleset: str
    status: CompetitionStatus
    requires_announced_starters: bool
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REGISTRY: tuple[CompetitionSpec, ...] = (
    CompetitionSpec(
        "mlb.regular_season",
        "MLB Regular Season",
        "baseball",
        "professional",
        "US",
        "MLB",
        "PRODUCTION",
        True,
    ),
    CompetitionSpec(
        "npb.regular_season",
        "NPB Regular Season",
        "baseball",
        "professional",
        "JP",
        "NPB",
        "PRODUCTION",
        True,
    ),
    CompetitionSpec(
        "wbsc.wbc",
        "World Baseball Classic",
        "baseball",
        "senior_national",
        "international",
        "WBSC-WBC",
        "RESEARCH_ONLY",
        True,
        "Promote only after competition-specific PIT and rules validation.",
    ),
    CompetitionSpec(
        "wbsc.premier12",
        "WBSC Premier12",
        "baseball",
        "senior_national",
        "international",
        "WBSC-PREMIER12",
        "RESEARCH_ONLY",
        True,
        "Promote only after competition-specific PIT and rules validation.",
    ),
    CompetitionSpec(
        "olympics.baseball",
        "Olympic Baseball",
        "baseball",
        "senior_national",
        "international",
        "OLYMPIC",
        "RESEARCH_ONLY",
        True,
        "Rules and roster eligibility differ by edition.",
    ),
    CompetitionSpec(
        "wbsc.u18",
        "WBSC U18",
        "baseball",
        "youth",
        "international",
        "WBSC-U18",
        "RESEARCH_ONLY",
        True,
    ),
    CompetitionSpec(
        "wbsc.u23",
        "WBSC U23",
        "baseball",
        "youth",
        "international",
        "WBSC-U23",
        "RESEARCH_ONLY",
        True,
    ),
    CompetitionSpec(
        "koshien.spring",
        "Senbatsu / Spring Koshien",
        "baseball",
        "high_school",
        "JP",
        "JHSBF",
        "RESEARCH_ONLY",
        True,
    ),
    CompetitionSpec(
        "koshien.summer",
        "Summer Koshien",
        "baseball",
        "high_school",
        "JP",
        "JHSBF",
        "RESEARCH_ONLY",
        True,
    ),
    CompetitionSpec(
        "japan.university",
        "Japanese University Baseball",
        "baseball",
        "university",
        "JP",
        "JUBF",
        "RESEARCH_ONLY",
        True,
    ),
)


def get_competition(competition_id: str) -> CompetitionSpec:
    for spec in REGISTRY:
        if spec.competition_id == competition_id:
            return spec
    raise KeyError(f"unknown competition_id: {competition_id}")


def production_competitions() -> tuple[CompetitionSpec, ...]:
    return tuple(s for s in REGISTRY if s.status == "PRODUCTION")


def validate_registry() -> None:
    ids = [s.competition_id for s in REGISTRY]
    if len(ids) != len(set(ids)):
        raise ValueError("competition_id values must be unique")
    for spec in REGISTRY:
        if not spec.competition_id or not spec.name or not spec.ruleset:
            raise ValueError(f"incomplete competition registry entry: {spec}")
        if spec.status == "PRODUCTION" and not spec.requires_announced_starters:
            raise ValueError(
                f"production competition must enforce announced starters: {spec.competition_id}"
            )
