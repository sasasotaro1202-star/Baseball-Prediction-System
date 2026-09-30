"""Autonomous free-source discovery and selection frontier.

The job searches public GitHub repositories and Hugging Face datasets, scores
candidates against competition-specific information needs, persists every newly
seen candidate, and selects bounded research candidates. Selection never grants
PIT validation or production eligibility.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from data.source_registry import SOURCES
from research.competition_catalog import SCOPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "auto_discovery_frontier.json"

GITHUB_SEARCH_QUERIES = (
    "baseball dataset play by play",
    "baseball pitch by pitch csv",
    "baseball statistics historical dataset",
    "baseball tournament dataset",
    "baseball scouting data",
)
FOCUS_LIMIT = int(os.getenv("BASEBALL_DISCOVERY_FOCUS_LIMIT", "12"))
RESULTS_PER_QUERY = int(os.getenv("BASEBALL_DISCOVERY_RESULTS_PER_QUERY", "8"))
MAX_TOTAL_RESULTS = int(os.getenv("BASEBALL_DISCOVERY_MAX_TOTAL_RESULTS", "120"))

def _get_json(url: str, *, token: str | None = None) -> Any:
    headers = {"User-Agent": "Baseball-Prediction-System-source-discovery/1.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))

def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()

def _source_key(platform: str, url: str) -> str:
    raw = platform + "|" + url
    return "AUTO_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def _query_for_scope(scope) -> str:
    label = _norm(scope.label)
    sid = _norm(scope.scope_id.replace("_", " "))
    extras = {
        "NPB": "japan npb", "MLB": "major league baseball", "KBO": "korea kbo",
        "CPBL": "taiwan cpbl", "NCAA_D1": "ncaa college baseball",
        "Japan_HighSchool": "koshien high school baseball japan",
        "Japan_University": "japan university baseball", "WBC": "world baseball classic",
        "WBSC_WomensBaseball": "women baseball wbsc",
        "LittleLeague_WorldSeries": "little league world series",
        "Japan_Independent": "japan independent baseball",
    }
    phrase = extras.get(scope.scope_id, f"{label} {sid}")
    return phrase + " dataset baseball"

def _rotating_focus() -> list[Any]:
    scopes = sorted(SCOPES, key=lambda s: (int(s.priority), s.scope_id))
    if not scopes:
        return []
    # Rotate by UTC day so the full frontier is explored over consecutive cycles.
    day = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    start = (day * max(1, FOCUS_LIMIT)) % len(scopes)
    ordered = scopes[start:] + scopes[:start]
    return ordered[: max(1, FOCUS_LIMIT)]

def _license_text(item: dict[str, Any]) -> str:
    lic = item.get("license") or item.get("cardData", {}).get("license") or ""
    if isinstance(lic, dict):
        lic = lic.get("spdx") or lic.get("name") or ""
    return str(lic)

def _score_candidate(*, scope, platform: str, name: str, description: str, url: str, license_name: str, stars: int = 0) -> tuple[float, dict[str, Any]]:
    text = _norm(" ".join((scope.label, scope.scope_id, name, description, url)))
    score = 0.0
    reasons = []
    tokens = [t for t in re.split(r"[^a-z0-9]+", _norm(scope.label + " " + scope.scope_id)) if len(t) >= 3]
    if any(t in text for t in tokens): score += 25; reasons.append("scope_match")
    if any(k in text for k in ("play by play", "pbp", "pitch by pitch", "pitch-level", "game log")): score += 20; reasons.append("game_granularity")
    if any(k in text for k in ("statcast", "tracking", "trackman", "hawkeye", "pitchcast")): score += 12; reasons.append("tracking_detail")
    if any(k in text for k in ("2010", "2015", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "historical", "archive")): score += 12; reasons.append("historical_depth_signal")
    if any(k in text for k in ("csv", "parquet", "json", "api", "dataset", "dataframe")): score += 10; reasons.append("machine_readable_signal")
    if any(k in text for k in ("timestamp", "available_at", "published_at", "publication", "release date", "archive date")): score += 15; reasons.append("pit_signal")
    if license_name: score += 8; reasons.append("license_declared")
    score += min(8.0, math.log1p(max(0, stars)) / 2.0)
    if any(k in text for k in ("paid api", "commercial only", "subscription required", "proprietary", "buy access")): score -= 35; reasons.append("cost_or_access_risk")
    if any(k in text for k in ("softball", "slowpitch", "fantasy only")): score -= 20; reasons.append("scope_risk")
    feature_map = {
        "schedule_identity": ("schedule", "game", "fixture", "score"),
        "starting_pitchers": ("starter", "pitcher", "probable", "starting"),
        "lineups": ("lineup", "roster", "batting order"),
        "batting": ("batting", "hitter", "batter"),
        "pitching": ("pitching", "pitcher", "strikeout", "earned run"),
        "fielding": ("fielding", "defense", "defensive"),
        "play_by_play": ("play by play", "pbp", "pitch by pitch", "game log"),
        "tracking": ("statcast", "tracking", "trackman", "hawkeye", "pitchcast", "velocity"),
        "weather": ("weather", "wind", "temperature"),
        "tournament_rules": ("tournament", "bracket", "playoff", "championship", "qualifier"),
    }
    inferred_features = sorted(
        feature for feature, keywords in feature_map.items()
        if any(k in text for k in keywords)
    )
    metadata = {
        "score": round(float(score), 3), "reasons": reasons,
        "features": inferred_features,
        "pit_status": "UNVERIFIED", "research_status": "DISCOVERED_UNVERIFIED",
        "production_eligible": False,
    }
    return float(score), metadata

def _github_candidates(query: str, token: str) -> list[dict[str, Any]]:
    url = "https://api.github.com/search/repositories?q=" + quote(query) + "&sort=updated&order=desc&per_page=" + str(RESULTS_PER_QUERY)
    payload = _get_json(url, token=token)
    return list(payload.get("items", []))

def _hf_candidates(query: str) -> list[dict[str, Any]]:
    url = "https://huggingface.co/api/datasets?search=" + quote(query) + "&limit=" + str(RESULTS_PER_QUERY)
    payload = _get_json(url)
    return payload if isinstance(payload, list) else []

def load_frontier() -> dict[str, Any]:
    if not OUT.exists():
        return {"schema_version": 1, "updated_at": None, "candidates": {}, "selection_history": []}
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("invalid auto discovery frontier schema")
    payload.setdefault("candidates", {})
    payload.setdefault("selection_history", [])
    return payload

def run() -> dict[str, Any]:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for bounded GitHub search")
    frontier = load_frontier()
    existing_registered = {str(s.source_id).lower() for s in SOURCES}
    focus = _rotating_focus()
    queries = [_query_for_scope(s) for s in focus] + list(GITHUB_SEARCH_QUERIES)
    seen_cycle: set[str] = set()
    discovered = 0
    failures = []
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for scope in focus:
        query = _query_for_scope(scope)
        for platform in ("github", "huggingface"):
            try:
                raw = _github_candidates(query, token) if platform == "github" else _hf_candidates(query)
            except Exception as exc:
                failures.append({"scope_id": scope.scope_id, "platform": platform, "error": f"{type(exc).__name__}:{exc}"})
                continue
            for item in raw:
                if len(seen_cycle) >= MAX_TOTAL_RESULTS:
                    break
                if platform == "github":
                    name = str(item.get("full_name") or item.get("name") or "")
                    desc = str(item.get("description") or "")
                    url = str(item.get("html_url") or "")
                    license_name = _license_text(item)
                    stars = int(item.get("stargazers_count") or 0)
                else:
                    name = str(item.get("id") or item.get("modelId") or "")
                    card = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
                    desc = str(item.get("description") or card.get("description") or "")
                    url = "https://huggingface.co/datasets/" + name if name else ""
                    license_name = _license_text(item)
                    stars = int(item.get("likes") or 0)
                if not name or not url:
                    continue
                key = _source_key(platform, url)
                seen_cycle.add(key)
                score, meta = _score_candidate(scope=scope, platform=platform, name=name, description=desc, url=url, license_name=license_name, stars=stars)
                record = frontier["candidates"].get(key, {})
                previous_scope_ids = list(record.get("scope_ids", []))
                record.update({
                    "source_id": key,
                    "scope_ids": sorted(set(previous_scope_ids + [scope.scope_id])),
                    "scope_id": scope.scope_id,
                    "scope_label": scope.label,
                    "platform": platform, "name": name, "description": desc, "url": url,
                    "license": license_name or None, "stars_or_likes": stars,
                    "first_seen": record.get("first_seen", now), "last_seen": now,
                    "discovery_queries": sorted(set(record.get("discovery_queries", []) + [query])),
                    "is_already_registered": key.lower() in existing_registered or url.lower() in {str(s.endpoint).lower() for s in SOURCES},
                    **meta,
                })
                if record["is_already_registered"]:
                    record["selection_status"] = "REGISTERED_REFERENCE"
                elif score >= 42:
                    record["selection_status"] = "SELECTED_RESEARCH_CANDIDATE"
                else:
                    record["selection_status"] = "DISCOVERED_QUEUE"
                frontier["candidates"][key] = record
                discovered += 1
    selected = [r for r in frontier["candidates"].values() if r.get("selection_status") == "SELECTED_RESEARCH_CANDIDATE"]
    selected.sort(key=lambda r: (-float(r.get("score", 0.0)), str(r.get("scope_id", "")), str(r.get("source_id", ""))))
    selected_by_scope = {}
    for r in selected:
        selected_by_scope.setdefault(str(r["scope_id"]), []).append(r["source_id"])

    for scope_id, ids in selected_by_scope.items():
        selected_by_scope[scope_id] = ids[:5]
    for r in frontier["candidates"].values():
        r["selected_for_next_research"] = r["source_id"] in set(sum(selected_by_scope.values(), []))
    frontier["updated_at"] = now
    frontier["last_run"] = {
        "run_at": now, "focus_scopes": [s.scope_id for s in focus], "query_count": len(queries),
        "discovered_records_touched": discovered, "selected_count": sum(len(v) for v in selected_by_scope.values()),
        "selected_by_scope": selected_by_scope, "failures": failures,
        "policy": {"free_only": True, "production_promotion": False, "unknown_pit": "FAIL_CLOSED"},
    }
    frontier["selection_history"].append(frontier["last_run"] )
    frontier["selection_history"] = frontier["selection_history"][-60:]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(frontier, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return frontier["last_run"]

if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))