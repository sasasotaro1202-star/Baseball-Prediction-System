"""Continuous competition-scope discovery for the baseball research system.

This is a discovery/research layer. It does not make a competition production
eligible and it does not generate predictions. It probes only explicitly
registered public sources, records reachability/content signals, and ranks the
next scope candidates so unsupported games can be reduced systematically.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.competition_registry import COMPETITIONS, CompetitionSpec
from data.source_registry import SOURCES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
_TIMEOUT = 15
_MAX_BYTES = 1_000_000
_DATE_RE = re.compile(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}\b")


def _probe(url: str) -> dict[str, Any]:
    if not url:
        return {"status": "NO_URL"}
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Baseball-Prediction-System/ScopeDiscovery"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            raw = response.read(_MAX_BYTES + 1)
            truncated = len(raw) > _MAX_BYTES
            raw = raw[:_MAX_BYTES]
            text = raw.decode("utf-8", "ignore")
            lower = text.lower()
            signals = {
                "schedule": int("schedule" in lower or "日程" in lower or "일정" in lower or "賽程" in lower or "calend" in lower),
                "results": int("result" in lower or "결과" in lower or "結果" in lower or "score" in lower),
                "game": int("game" in lower or "경기" in lower or "試合" in lower or "juego" in lower),
                "upcoming": int("upcoming" in lower or "未開始" in lower or "next" in lower or "upcoming games" in lower),
                "date_tokens": len(_DATE_RE.findall(text)),
            }
            return {
                "status": "REACHABLE",
                "http_status": int(getattr(response, "status", 200)),
                "bytes": len(raw),
                "truncated": truncated,
                "signals": signals,
            }
    except Exception as exc:
        return {
            "status": "UNREACHABLE",
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        }


def _priority_score(spec: CompetitionSpec, probe: dict[str, Any]) -> float:
    signals = probe.get("signals", {})
    reach = 1.0 if probe.get("status") == "REACHABLE" else 0.0
    schedule = float(signals.get("schedule", 0))
    game = float(signals.get("game", 0))
    upcoming = float(signals.get("upcoming", 0))
    dates = min(float(signals.get("date_tokens", 0)) / 5.0, 1.0)
    # Higher score means earlier research attention. Discovery priority is only
    # a prior; source reachability and schedule signals can outrank it.
    return 100.0 / max(1, spec.discovery_priority) + 3.0 * reach + 2.0 * schedule + 2.0 * game + 1.0 * upcoming + dates


def discover_scope() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    # The competition registry is intentionally not the sole discovery boundary.
    # Any league/source family newly added to source_registry becomes an explicit
    # frontier candidate until a competition contract is registered.
    registered_ids = {spec.competition_id for spec in COMPETITIONS}
    registered_names = {spec.name for spec in COMPETITIONS}
    frontier: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        key = source.league
        if key in {"NPB+MLB", "MLB+Historical", "Cross-Level-Research", "WBC+PlayerPrior",
                   "WBSC-U12-U15-U18-U23", "Japan-U12-U15-U18"}:
            continue
        if key in registered_ids or key in registered_names:
            continue
        item = frontier.setdefault(key, {
            "competition_id": key,
            "name": key,
            "status": "UNREGISTERED_FRONTIER",
            "source_ids": [],
            "features": [],
            "urls": [],
        })
        item["source_ids"].append(source.source_id)
        item["features"].append(source.feature)
        if str(source.endpoint).startswith(("http://", "https://")):
            item["urls"].append(source.endpoint)
    for item in frontier.values():
        item["source_ids"] = sorted(set(item["source_ids"]))
        item["features"] = sorted(set(item["features"]))
        item["urls"] = sorted(set(item["urls"]))
        item["next_stage"] = "COMPETITION_CONTRACT_AND_DATA_PIT_VALIDATION"
        item["production_eligible"] = False
    discovered_frontier = sorted(frontier.values(), key=lambda x: (len(x["source_ids"]) * -1, x["competition_id"]))

    for spec in COMPETITIONS:
        if not spec.discovery_url:
            continue
        probe = _probe(spec.discovery_url)
        row = asdict(spec)
        row["probe"] = probe
        row["discovery_score"] = _priority_score(spec, probe)
        row["production_eligible"] = bool(
            spec.status == "PRODUCTION_ELIGIBLE" and probe.get("status") == "REACHABLE"
        )
        # No automatic promotion: a reachable source is only a discovery signal.
        row["next_stage"] = (
            "DATA_PIT_VALIDATION" if probe.get("status") == "REACHABLE" else "SOURCE_RECOVERY"
        )
        rows.append(row)

    rows.sort(key=lambda x: (-float(x["discovery_score"]), x["competition_id"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTED",
        "policy": {
            "production_promotion": "NEVER_BY_DISCOVERY",
            "unknown_pit": "FAIL_CLOSED",
            "research_scope_expansion": "CONTINUOUS",
            "free_public_sources_only": True,
        },
        "candidates": rows,
        "unregistered_frontier": discovered_frontier,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "scope_discovery.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    payload = discover_scope()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
