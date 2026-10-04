"""PIT-aware starter/lineup announcement records and eligibility gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

from data.competition_registry import get as get_competition
from core.pit_evidence import _is_official_source

_ALLOWED_LINEUP_STATUS = {"CONFIRMED", "UNCONFIRMED", "UNAVAILABLE", "UNVERIFIABLE"}


def _dt(value: str) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("availability timestamps must be timezone-aware")
    return ts


@dataclass(frozen=True)
class AvailabilityRecord:
    event_id: str
    league: str
    home_team: str
    away_team: str
    home_starter: str | None
    away_starter: str | None
    home_starter_announced_at: str | None
    away_starter_announced_at: str | None
    lineup_status: str
    lineup_announced_at: str | None
    source: str
    retrieved_at: str
    prediction_cutoff: str
    event_start_at: str | None = None
    home_starter_published_at: str | None = None
    away_starter_published_at: str | None = None
    home_starter_available_at: str | None = None
    away_starter_available_at: str | None = None
    revision_time: str | None = None

    def validate(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        # Keep the legacy field name for compatibility, but validate it against
        # the canonical registry so newly registered competitions are not
        # rejected by an obsolete NPB/MLB-only hardcode. Registry membership
        # alone never grants production eligibility.
        try:
            get_competition(self.league)
        except KeyError as exc:
            raise ValueError(f"unknown competition_id/league: {self.league}") from exc

        if not self.home_team or not self.away_team:
            raise ValueError("home_team and away_team are required")
        if self.lineup_status.upper() not in _ALLOWED_LINEUP_STATUS:
            raise ValueError("invalid lineup_status")

        cutoff = _dt(self.prediction_cutoff)
        retrieved = _dt(self.retrieved_at)
        if retrieved > cutoff:
            raise ValueError("retrieved_at is after prediction cutoff")
        if self.event_start_at:
            _dt(self.event_start_at)
        if self.revision_time:
            revision_ts = _dt(self.revision_time)
            if revision_ts > cutoff:
                raise ValueError("revision_time is after prediction cutoff")
            if revision_ts > retrieved:
                raise ValueError("revision_time is after source retrieval")

        for starter, announced_at, published_at, available_at, side in (
            (
                self.home_starter,
                self.home_starter_announced_at,
                self.home_starter_published_at,
                self.home_starter_available_at,
                "home",
            ),
            (
                self.away_starter,
                self.away_starter_announced_at,
                self.away_starter_published_at,
                self.away_starter_available_at,
                "away",
            ),
        ):
            if starter and not announced_at:
                raise ValueError(f"{side} starter has unknown announcement timestamp")
            if published_at and not announced_at:
                raise ValueError(f"{side} starter has publication timestamp without announcement timestamp")
            if announced_at:
                announced_ts = _dt(announced_at)
                if announced_ts > cutoff:
                    raise ValueError(f"{side} starter was announced after prediction cutoff")
                if announced_ts > retrieved:
                    raise ValueError(f"{side} starter announcement is after source retrieval")
                effective_available = _dt(available_at) if available_at else announced_ts
                if effective_available > cutoff:
                    raise ValueError(f"{side} starter availability is after prediction cutoff")
                if effective_available > retrieved:
                    raise ValueError(f"{side} starter availability is after source retrieval")
                if published_at:
                    published_ts = _dt(published_at)
                    if published_ts > announced_ts:
                        raise ValueError(f"{side} starter publication is after announcement time")
                    if published_ts > cutoff:
                        raise ValueError(f"{side} starter publication is after prediction cutoff")
            elif available_at:
                available_ts = _dt(available_at)
                if available_ts > cutoff or available_ts > retrieved:
                    raise ValueError(f"{side} starter availability is not PIT-safe")

        if self.lineup_announced_at:
            ts = _dt(self.lineup_announced_at)
            if ts > cutoff or ts > retrieved:
                raise ValueError("lineup announcement is not PIT-safe")
        if self.lineup_status.upper() == "CONFIRMED" and not self.lineup_announced_at:
            raise ValueError("CONFIRMED lineup requires lineup_announced_at")


def from_mapping(row: Mapping[str, Any]) -> AvailabilityRecord:
    record = AvailabilityRecord(
        event_id=str(row["event_id"]), league=str(row["league"]),
        home_team=str(row["home_team"]), away_team=str(row["away_team"]),
        home_starter=row.get("home_starter"), away_starter=row.get("away_starter"),
        home_starter_announced_at=row.get("home_starter_announced_at"),
        away_starter_announced_at=row.get("away_starter_announced_at"),
        lineup_status=str(row.get("lineup_status", "UNVERIFIABLE")).upper(),
        lineup_announced_at=row.get("lineup_announced_at"),
        source=str(row.get("source", "UNVERIFIABLE")),
        retrieved_at=str(row["retrieved_at"]), prediction_cutoff=str(row["prediction_cutoff"]),
        event_start_at=row.get("event_start_at"),
    )
    record.validate()
    return record


def prediction_eligible(record: AvailabilityRecord) -> tuple[bool, list[str]]:
    record.validate()
    spec = get_competition(record.league)
    reasons: list[str] = []
    if not record.home_starter:
        reasons.append("home_starter_not_confirmed")
    elif not record.home_starter_announced_at:
        reasons.append("home_starter_announcement_time_unverified")
    if not record.away_starter:
        reasons.append("away_starter_not_confirmed")
    elif not record.away_starter_announced_at:
        reasons.append("away_starter_announcement_time_unverified")
    return (not reasons, reasons)


def production_prediction_eligible(record: AvailabilityRecord) -> tuple[bool, list[str]]:
    """Apply both PIT starter eligibility and the canonical production gate.

    Registry membership is intentionally insufficient: research-only
    competitions may pass the PIT starter checks while remaining blocked from
    production until their competition-specific evidence and OOS/holdout gates
    have been promoted.
    """
    ok, reasons = prediction_eligible(record)
    if not record.event_start_at:
        reasons.append("event_start_time_missing")
    else:
        cutoff = _dt(record.prediction_cutoff)
        event_start = _dt(record.event_start_at)
        if event_start <= cutoff:
            reasons.append("event_already_started_or_not_future")
    if not _is_official_source(record.source):
        reasons.append("starter_source_not_official")
    spec = get_competition(record.league)
    if not spec.status == "PRODUCTION_ELIGIBLE":
        reasons.append("competition_not_production_eligible")
    return (not reasons, reasons)


def as_dict(record: AvailabilityRecord) -> dict[str, Any]:
    return asdict(record)
