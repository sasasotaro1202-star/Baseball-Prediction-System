"""Competition-aware daily scope routing.

Inventory, research activity, and production eligibility are separate states.
Discovery must not promote a competition to production.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "global_baseball_competition_registry.json"
RUNTIME = ROOT / "config" / "current_production_runtime.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid JSON object: {path}")
    return payload


def build_scope() -> dict[str, Any]:
    registry = _load(REGISTRY)
    runtime = _load(RUNTIME)
    entries = registry.get("competitions") or []
    runtimes = runtime.get("runtimes") or {}
    active = set((registry.get("execution_scope") or {}).get("active_competitions") or [])

    rows: list[dict[str, Any]] = []
    for entry in entries:
        cid = str(entry.get("competition_id", "")).strip()
        if not cid:
            continue
        upper = cid.upper()
        rt = runtimes.get(upper) or {}
        formal = str(rt.get("formal_adoption_status", "")).upper()
        is_current_production = formal == "CURRENT_PRODUCTION" and bool(
            str(rt.get("entrypoint", "")).strip()
        )
        is_active = cid in active
        if is_current_production:
            state = "CURRENT_PRODUCTION"
        elif is_active and str(entry.get("implementation_status", "")).upper() == "IMPLEMENTED":
            state = "RESEARCH_ACTIVE"
        elif is_active:
            state = "RESEARCH_DISCOVERY"
        else:
            state = "DEFERRED"
        rows.append(
            {
                "competition_id": cid,
                "name": entry.get("name", cid),
                "implementation_status": entry.get("implementation_status", "UNKNOWN"),
                "registry_execution_scope": (
                    "ACTIVE" if is_active else entry.get("execution_scope", "DEFERRED")
                ),
                "current_production": is_current_production,
                "runtime_status": formal or "UNREGISTERED",
                "state": state,
                "production_entrypoint": rt.get("entrypoint", ""),
                "production_model_version": rt.get("model_version", ""),
            }
        )

    return {
        "schema_version": "baseball-competition-scope-v1",
        "policy": {
            "production_promotion": "NEVER_BY_DISCOVERY",
            "unknown_pit": "FAIL_CLOSED",
            "research_isolation": True,
        },
        "current_production": [r["competition_id"] for r in rows if r["state"] == "CURRENT_PRODUCTION"],
        "research_active": [r["competition_id"] for r in rows if r["state"] == "RESEARCH_ACTIVE"],
        "research_discovery": [r["competition_id"] for r in rows if r["state"] == "RESEARCH_DISCOVERY"],
        "deferred": [r["competition_id"] for r in rows if r["state"] == "DEFERRED"],
        "competitions": rows,
    }


def main() -> int:
    print(json.dumps(build_scope(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
