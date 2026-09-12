"""Historical market-line contract for PIT-safe Low/High evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

_ALLOWED = {"KNOWN", "MISSING", "UNAVAILABLE", "UNVERIFIABLE"}


def _dt(value: str) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("market timestamps must be timezone-aware")
    return ts


@dataclass(frozen=True)
class TotalRunsLine:
    event_id: str
    league: str
    line: float | None
    source: str
    observed_at: str
    available_at: str | None
    status: str = "KNOWN"

    def validate(self) -> None:
        if not self.event_id or self.league not in {"NPB", "MLB"}:
            raise ValueError("event_id and league (NPB/MLB) are required")
        if self.status not in _ALLOWED:
            raise ValueError(f"invalid line status: {self.status}")
        observed = _dt(self.observed_at)
        if self.status == "KNOWN" and self.line is None:
            raise ValueError("KNOWN market line requires a numeric line")
        if self.line is not None:
            if self.line < 0 or self.line > 100:
                raise ValueError("total runs line is outside a sane range")
            # Standard baseball totals are generally integer or half-point lines.
            if abs(self.line * 2 - round(self.line * 2)) > 1e-9:
                raise ValueError("market line must be an integer or half-point")
        if self.available_at is not None:
            available = _dt(self.available_at)
            if available < observed:
                raise ValueError("available_at cannot precede observed_at")
        elif self.status == "KNOWN":
            raise ValueError("KNOWN market line requires available_at")

    def pit_usable(self, cutoff: str) -> bool:
        self.validate()
        if self.status != "KNOWN" or self.line is None or not self.available_at:
            return False
        cutoff_ts = _dt(cutoff)
        return _dt(self.observed_at) <= cutoff_ts and _dt(self.available_at) <= cutoff_ts


def low_high_threshold(line: float) -> tuple[float, float]:
    """Return the integer Low/High boundary for a standard baseball total."""
    line = float(line)
    if abs(line * 2 - round(line * 2)) > 1e-9 or line < 0:
        raise ValueError("market line must be an integer or half-point")
    if line.is_integer():
        cutoff = int(line)
    else:
        cutoff = int(line + 0.5)
    return float(cutoff), float(cutoff + 1)


def classify_total(total_runs: int, line: float) -> str:
    line = float(line)
    if line.is_integer():
        if total_runs == int(line):
            return "PUSH"
        return "LOW" if total_runs < line else "HIGH"
    return "LOW" if total_runs < line else "HIGH"


def from_mapping(row: Mapping[str, Any]) -> TotalRunsLine:
    line = row.get("line")
    obj = TotalRunsLine(
        event_id=str(row["event_id"]), league=str(row["league"]),
        line=None if line in (None, "") else float(line),
        source=str(row.get("source", "UNVERIFIABLE")),
        observed_at=str(row["observed_at"]),
        available_at=row.get("available_at"), status=str(row.get("status", "KNOWN")).upper(),
    )
    obj.validate()
    return obj


def pit_usable(line: TotalRunsLine, cutoff: str) -> bool:
    return line.pit_usable(cutoff)
