"""Fail-closed validation for external research-tool candidates.

This registry is deliberately separate from the baseball production source
registry. A research tool can discover, fetch, browse, audit, or structure
information, but it is never itself prediction evidence and never grants
production eligibility.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "research" / "external_research_tools.json"
ALLOWED_STATUS = {"ADOPT_CANDIDATE", "RESEARCH_CANDIDATE", "HOLD", "REJECT"}
REQUIRED_EVIDENCE_BASE = {"SOURCE_VERIFIED", "COST_CHECK", "LOCAL_REPRODUCTION"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("external research tool registry must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported external research tool registry schema")
    if payload.get("auto_install") is not False:
        raise ValueError("automatic installation must remain disabled")
    if payload.get("auto_promotion") is not False:
        raise ValueError("automatic promotion must remain disabled")
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("external research tool registry has no tools")
    return payload


def validate_tool(tool: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "tool_id",
        "repository",
        "repository_url",
        "pinned_ref",
        "license",
        "status",
        "purpose",
        "cost_policy",
        "security_policy",
        "pit_policy",
        "required_evidence",
    ):
        if key not in tool:
            errors.append(f"missing:{key}")
    tool_id = str(tool.get("tool_id") or "")
    if not tool_id:
        errors.append("empty:tool_id")
    if not SHA_RE.fullmatch(str(tool.get("pinned_ref") or "").lower()):
        errors.append(f"invalid_pinned_ref:{tool_id}")
    if str(tool.get("status")) not in ALLOWED_STATUS:
        errors.append(f"invalid_status:{tool_id}")
    if tool.get("production_eligible") is not False:
        errors.append(f"production_eligible_must_be_false:{tool_id}")
    if tool.get("auto_install") is not False:
        errors.append(f"auto_install_must_be_false:{tool_id}")
    if tool.get("auto_promotion") is not False:
        errors.append(f"auto_promotion_must_be_false:{tool_id}")
    required = set(tool.get("required_evidence") or [])
    missing = sorted(REQUIRED_EVIDENCE_BASE - required)
    if missing:
        errors.append(f"required_evidence_missing:{tool_id}:{','.join(missing)}")
    if "PIT_CHECK" not in required and str(tool.get("purpose", "")).find("security") < 0:
        errors.append(f"pit_check_missing:{tool_id}")
    if "unknown" in str(tool.get("cost_policy", "")).lower() and str(tool.get("status")) != "HOLD":
        errors.append(f"unknown_cost_not_hold:{tool_id}")
    return errors


def validate_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for tool in payload["tools"]:
        if not isinstance(tool, dict):
            errors.append("tool_entry_not_object")
            continue
        tool_id = str(tool.get("tool_id") or "")
        if tool_id in seen:
            errors.append(f"duplicate_tool_id:{tool_id}")
        seen.add(tool_id)
        errors.extend(validate_tool(tool))
    return errors


def selected_pilot_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidates eligible for local research pilots, never production."""
    return [
        tool for tool in payload["tools"]
        if str(tool.get("status")) in {"ADOPT_CANDIDATE", "RESEARCH_CANDIDATE"}
        and tool.get("production_eligible") is False
        and tool.get("auto_promotion") is False
    ]


def main() -> int:
    payload = load_registry()
    errors = validate_registry(payload)
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    pilots = selected_pilot_tools(payload)
    print(json.dumps({
        "status": "PASS",
        "schema_version": payload["schema_version"],
        "pilot_count": len(pilots),
        "pilot_tools": [tool["tool_id"] for tool in pilots],
        "production_eligible": False,
        "auto_install": False,
        "auto_promotion": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
