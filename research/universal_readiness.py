"""Universal readiness state machine for baseball sources and scopes.

Registration is not evidence of usability. Each source and scope is tracked
through explicit gates so unresolved acquisition/PIT/OOS work remains visible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from data.source_registry import SOURCES
from research.competition_catalog import scopes
from research.universal_source_matrix import application_matrix
from research.universal_adapter_registry import adapter_metadata


STATES = (
    "REGISTERED",
    "ADAPTER",
    "PIT",
    "OOS",
    "PRODUCTION",
    "HOLD",
    "REJECTED",
)


@dataclass(frozen=True)
class SourceReadiness:
    source_id: str
    registered: bool
    adapter_status: str
    pit_status: str
    oos_status: str
    production_status: str

    @property
    def state(self) -> str:
        if self.production_status == "PASS":
            return "PRODUCTION"
        if self.oos_status == "PASS":
            return "OOS"
        if self.pit_status == "PASS":
            return "PIT"
        if self.adapter_status == "PASS":
            return "ADAPTER"
        if any(x == "FAIL" for x in (self.adapter_status, self.pit_status, self.oos_status, self.production_status)):
            return "HOLD"
        return "REGISTERED"


# Explicit evidence-backed statuses for the limited sources already wired into
# this repository. Everything else remains UNVERIFIED until its own evidence is
# recorded. This is intentionally conservative.
KNOWN_STATUS: dict[str, dict[str, str]] = {
    "statcast": {"adapter_status": "PASS", "pit_status": "UNVERIFIED", "oos_status": "UNVERIFIED", "production_status": "FAIL"},
    "mlb_milb_statcast": {"adapter_status": "PASS", "pit_status": "UNVERIFIED", "oos_status": "UNVERIFIED", "production_status": "FAIL"},
    "npb_hawkeye_npbplus": {"adapter_status": "UNWIRED", "pit_status": "UNVERIFIED", "oos_status": "UNVERIFIED", "production_status": "FAIL"},
    "npb_public_spaia_pbp": {"adapter_status": "PASS", "pit_status": "UNVERIFIED", "oos_status": "UNVERIFIED", "production_status": "FAIL"},
    "wocchi09_npb_data": {"adapter_status": "UNWIRED", "pit_status": "UNVERIFIED", "oos_status": "UNVERIFIED", "production_status": "FAIL"},
    "armstjc_npb_repository": {"adapter_status": "PASS", "pit_status": "UNVERIFIED", "oos_status": "UNVERIFIED", "production_status": "FAIL"},
    "weather": {"adapter_status": "PASS", "pit_status": "UNVERIFIED", "oos_status": "UNVERIFIED", "production_status": "FAIL"},
}


def source_readiness(source_id: str) -> SourceReadiness:
    registered = source_id in {s.source_id for s in SOURCES}
    if not registered:
        raise KeyError(f"unknown source_id: {source_id}")
    values = {
        "adapter_status": "UNWIRED",
        "pit_status": "UNVERIFIED",
        "oos_status": "UNVERIFIED",
        "production_status": "FAIL",
    }
    values.update(KNOWN_STATUS.get(source_id, {}))
    return SourceReadiness(source_id=source_id, registered=True, **values)


def all_source_readiness() -> list[SourceReadiness]:
    return [source_readiness(s.source_id) for s in SOURCES]


def scope_readiness(scope_id: str) -> dict[str, Any]:
    matching = [r for r in application_matrix() if r["scope_id"] == scope_id]
    if not matching:
        raise KeyError(f"unknown scope_id: {scope_id}")
    sources = [source_readiness(r["source_id"]) for r in matching]
    return {
        "scope_id": scope_id,
        "sources": len(sources),
        "adapter_pass": sum(x.adapter_status == "PASS" for x in sources),
        "adapter_implemented": sum(bool(adapter_metadata(x.source_id)["implemented"]) for x in sources),
        "pit_pass": sum(x.pit_status == "PASS" for x in sources),
        "oos_pass": sum(x.oos_status == "PASS" for x in sources),
        "production_pass": sum(x.production_status == "PASS" for x in sources),
        "blocking_states": sorted({x.state for x in sources if x.state not in {"PRODUCTION", "OOS", "PIT", "ADAPTER"}}),
    }


def readiness_report() -> dict[str, Any]:
    source_rows = all_source_readiness()
    scope_rows = [scope_readiness(s.scope_id) for s in scopes()]
    return {
        "source_count": len(source_rows),
        "scope_count": len(scope_rows),
        "source_states": {
            state: sum(r.state == state for r in source_rows)
            for state in STATES
        },
        "sources": [
            asdict(r) | {"state": r.state, "adapter": adapter_metadata(r.source_id)}
            for r in source_rows
        ],
        "scopes": scope_rows,
        "promotion_rule": "No source/scope advances without explicit adapter, PIT, chronological OOS, and production evidence.",
    }


def main() -> int:
    import json
    print(json.dumps(readiness_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
