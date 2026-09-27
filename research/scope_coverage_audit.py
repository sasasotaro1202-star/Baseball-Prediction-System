"""Audit baseball scope coverage as an explicit discovery-to-production funnel.

The audit measures where each competition/scope currently stops:
CATALOG -> REGISTRY -> SOURCE -> ADAPTER -> DISCOVERY -> PIT -> OOS -> PRODUCTION.
Missing evidence is visible and fail-closed; no stage infers the next stage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.competition_registry import COMPETITIONS
from data.source_registry import SOURCES
from research.competition_catalog import scopes
from research.universal_adapter_registry import adapter_metadata
from research.universal_readiness import scope_readiness

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_FILE = ROOT / "results" / "scope_discovery.json"


def _discovery_indexes() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not DISCOVERY_FILE.exists() or DISCOVERY_FILE.stat().st_size == 0:
        return {}, {}
    try:
        payload = json.loads(DISCOVERY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    registered = {
        str(row.get("competition_id")): row
        for row in payload.get("candidates", [])
        if row.get("competition_id")
    }
    catalog = {
        str(row.get("scope_id")): row
        for row in payload.get("catalog_frontier", [])
        if row.get("scope_id")
    }
    return registered, catalog


def _pit_counts() -> dict[str, int]:
    path = ROOT / "data" / "pit" / "source_snapshots.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        league = str(row.get("league", "")).strip()
        if not league:
            continue
        if str(row.get("status", "KNOWN")).upper() != "KNOWN":
            continue
        if not row.get("available_at") or not row.get("prediction_cutoff"):
            continue
        out[league] = out.get(league, 0) + 1
    return out


def _stage(*, registry: bool, source_count: int, adapter_count: int, discovered: bool,
           pit_count: int, oos_pass: bool, production_pass: bool) -> str:
    if production_pass:
        return "PRODUCTION"
    if oos_pass:
        return "OOS"
    if pit_count > 0:
        return "PIT"
    if adapter_count > 0:
        return "ADAPTER"
    if discovered:
        return "DISCOVERED"
    if source_count > 0 and registry:
        return "SOURCE_REGISTERED"
    if registry:
        return "REGISTERED"
    return "CATALOG_ONLY"


def _frontier_game_metrics() -> dict[str, dict[str, int]]:
    root = ROOT / "results" / "24h"
    totals: dict[str, dict[str, int]] = {}
    if not root.exists():
        return totals
    for path in root.glob("cycle_*/*.json"):
        name = path.stem
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = {
            "kbo": "KBO",
            "cpbl": "CPBL",
            "espn_mlb": "MLB",
            "espn_ncaa": "NCAA_D1_BASEBALL",
            "iblj": "JAPAN_INDEPENDENT",
            "bcl": "JAPAN_INDEPENDENT",
        }.get(name)
        if not key:
            continue
        row = totals.setdefault(key, {
            "observations": 0,
            "games_discovered": 0,
            "event_count": 0,
            "token_count": 0,
            "deferred_or_unparsed": 0,
            "source_failures": 0,
        })
        row["observations"] += 1
        row["games_discovered"] += int(obj.get("game_count", 0) or 0)
        row["event_count"] += int(obj.get("event_count", 0) or 0)
        row["token_count"] += int(obj.get("game_token_count", 0) or 0)
        if str(obj.get("availability_status", "")).startswith("DISCOVERED_NOT"):
            row["deferred_or_unparsed"] += int(obj.get("game_token_count", 0) or obj.get("game_count", 0) or obj.get("event_count", 0) or 1)
        if str(obj.get("status", "")).upper() not in {"EXECUTED", "PASS", "OK"}:
            row["source_failures"] += 1
    return totals


def audit() -> dict[str, Any]:
    registry_by_id = {x.competition_id: x for x in COMPETITIONS}
    sources_by_id = {x.source_id: x for x in SOURCES}
    discovery, catalog_discovery = _discovery_indexes()
    pit_counts = _pit_counts()
    frontier_metrics = _frontier_game_metrics()

    rows: list[dict[str, Any]] = []
    for scope in scopes():
        reg = registry_by_id.get(scope.scope_id)
        applied_source_ids = [sid for sid in scope.source_ids if sid in sources_by_id]
        adapter_sources = [
            sid for sid in applied_source_ids
            if adapter_metadata(sid)["implemented"]
        ]
        discovery_row = discovery.get(scope.scope_id, {})
        catalog_row = catalog_discovery.get(scope.scope_id, {})
        pit_count = pit_counts.get(scope.scope_id, 0)

        oos_pass = False
        production_pass = False
        if reg:
            try:
                readiness = scope_readiness(scope.scope_id)
                # universal_readiness is a source-level readiness signal; only
                # PASS values are counted and no inferred PASS is allowed.
                oos_pass = readiness["oos_pass"] == readiness["sources"] and readiness["sources"] > 0
                production_pass = readiness["production_pass"] == readiness["sources"] and readiness["sources"] > 0
            except Exception:
                oos_pass = production_pass = False

        stage = _stage(
            registry=bool(reg),
            source_count=len(applied_source_ids),
            adapter_count=len(adapter_sources),
            discovered=bool(discovery_row) or bool(catalog_row),
            pit_count=pit_count,
            oos_pass=oos_pass,
            production_pass=production_pass,
        )
        rows.append({
            "scope_id": scope.scope_id,
            "label": scope.label,
            "registry_registered": bool(reg),
            "registry_status": getattr(reg, "status", "NOT_REGISTERED"),
            "source_count": len(applied_source_ids),
            "source_ids": applied_source_ids,
            "adapter_count": len(adapter_sources),
            "adapter_source_ids": adapter_sources,
            "discovered": bool(discovery_row) or bool(catalog_row),
            "discovery_reachable_source_count": int(
                discovery_row.get("probe", {}).get("status") == "REACHABLE"
            ) if discovery_row else int(catalog_row.get("reachable_source_count", 0)),
            "pit_snapshot_rows": pit_count,
            "frontier_game_metrics": frontier_metrics.get(scope.scope_id, {
                "observations": 0,
                "games_discovered": 0,
                "event_count": 0,
                "token_count": 0,
                "deferred_or_unparsed": 0,
                "source_failures": 0,
            }),
            "oos_pass": oos_pass,
            "production_pass": production_pass,
            "stage": stage,
            "next_action": {
                "CATALOG_ONLY": "REGISTER_COMPETITION_AND_SOURCE_CONTRACT",
                "REGISTERED": "WIRE_SOURCE_AND_DISCOVERY",
                "SOURCE_REGISTERED": "IMPLEMENT_OR_REPAIR_ADAPTER",
                "DISCOVERED": "VALIDATE_DATA_AND_PIT",
                "ADAPTER": "PROVE_HISTORICAL_PIT",
                "PIT": "RUN_CHRONOLOGICAL_OOS",
                "OOS": "RUN_PRODUCTION_GATES",
                "PRODUCTION": "MONITOR_AND_EXPAND",
            }.get(stage, "HOLD_AND_INVESTIGATE"),
        })

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1

    return {
        "schema_version": 1,
        "status": "AUDIT_COMPLETE",
        "scope_count": len(rows),
        "stage_counts": dict(sorted(counts.items())),
        "frontier_game_metrics": frontier_metrics,
        "scopes": rows,
        "policy": {
            "coverage_goal": "continuously increase supported/safely researchable game scope",
            "unknown_pit": "FAIL_CLOSED",
            "missing_data_is_not_zero": True,
            "promotion_by_coverage_audit": False,
        },
    }


def main() -> int:
    payload = audit()
    out = ROOT / "results" / "scope_coverage_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
