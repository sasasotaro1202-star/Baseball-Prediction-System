#!/usr/bin/env python3
"""Research-only X public-post collector with conservative PIT semantics.

This module is deliberately outside the production feature path.  It collects
public X posts for future offline research and records the earliest time at
which THIS pipeline observed each post.  A post's created_at is event time,
not proof that the post was available to a historical backtest at that time.
Therefore historical replay may use a collected observation only at/after
observed_available_at unless independent historical availability evidence is
present.

The collector uses X API v2 Recent Search, which is limited to the most recent
7 days for the broadly available search tier.  Full-archive search is not used
here because it requires higher access.  See docs.x.com for current access and
policy requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
X_DIR = ROOT / "data" / "x"
RAW_LOG = X_DIR / "raw_posts.jsonl"
PIT_LOG = ROOT / "data" / "pit" / "x_source_snapshots.jsonl"
RUN_LOG = X_DIR / "acquisition_runs.jsonl"
BASE_URL = "https://api.x.com/2/tweets/search/recent"
MAX_LOOKBACK_DAYS = 7
DEFAULT_MAX_RESULTS = 100
TIMEOUT = int(os.getenv("X_API_TIMEOUT", "20"))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def payload_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_window(start: datetime, end: datetime, now: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("X collection timestamps must be timezone-aware")
    if end <= start:
        raise ValueError("end_time must be after start_time")
    if end > now + timedelta(minutes=1):
        raise ValueError("end_time cannot be in the future")
    if start < now - timedelta(days=MAX_LOOKBACK_DAYS):
        raise ValueError("recent-search window exceeds the broadly available 7-day limit")


def search_recent(token: str, query: str, start: datetime, end: datetime, max_results: int) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "start_time": iso(start),
        "end_time": iso(end),
        "max_results": max(10, min(int(max_results), 100)),
        "tweet.fields": "id,text,created_at,author_id,lang,conversation_id,public_metrics,entities,context_annotations,referenced_tweets",
        "expansions": "author_id",
        "user.fields": "id,name,username,verified,public_metrics",
    }
    response = requests.get(
        BASE_URL,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=(5, TIMEOUT),
    )
    response.raise_for_status()
    body = response.json()
    posts = body.get("data") or []
    includes = body.get("includes") or {}
    users = {str(u.get("id")): u for u in (includes.get("users") or []) if u.get("id")}
    observed_at = iso(now_utc())
    out: list[dict[str, Any]] = []
    for post in posts:
        post_id = str(post.get("id") or "")
        if not post_id:
            continue
        row = {
            "post_id": post_id,
            "query": query,
            "created_at": post.get("created_at"),
            "author_id": post.get("author_id"),
            "author": users.get(str(post.get("author_id"))) if post.get("author_id") else None,
            "lang": post.get("lang"),
            "text": post.get("text", ""),
            "conversation_id": post.get("conversation_id"),
            "public_metrics": post.get("public_metrics") or {},
            "entities": post.get("entities") or {},
            "context_annotations": post.get("context_annotations") or [],
            "referenced_tweets": post.get("referenced_tweets") or [],
            # Conservative PIT boundary: this pipeline only proves availability
            # at the moment it successfully retrieved the post.
            "observed_available_at": observed_at,
            "retrieved_at": observed_at,
            "prediction_cutoff": observed_at,
            "source": "X_API_V2_RECENT_SEARCH",
        }
        row["payload_hash"] = payload_hash(row)
        append_jsonl(RAW_LOG, row)
        append_jsonl(PIT_LOG, {
            "event_id": f"X:{post_id}",
            "league": "NPB+MLB",
            "entity_type": "public_post",
            "entity_id": post_id,
            "source": "X_API_V2_RECENT_SEARCH",
            "source_timestamp": post.get("created_at"),
            "retrieved_at": observed_at,
            "available_at": observed_at,
            "prediction_cutoff": observed_at,
            "payload_hash": row["payload_hash"],
            "status": "KNOWN",
            "historical_availability_proven": False,
        })
        out.append(row)
    return out


def collect(queries: list[str], start: datetime, end: datetime, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    token = os.getenv("X_BEARER_TOKEN", "").strip()
    if not token:
        raise RuntimeError("X_BEARER_TOKEN is required; no collection is attempted without credentials")
    validate_window(start, end, now_utc())
    totals = 0
    per_query: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    for query in queries:
        query = query.strip()
        if not query:
            continue
        try:
            rows = search_recent(token, query, start, end, max_results)
            per_query[query] = len(rows)
            totals += len(rows)
        except Exception as exc:
            errors.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "started_at": iso(now_utc()),
        "window_start": iso(start),
        "window_end": iso(end),
        "queries": queries,
        "posts_collected": totals,
        "per_query": per_query,
        "errors": errors,
        "source": "X_API_V2_RECENT_SEARCH",
        "historical_backtest_eligible": False,
        "reason": "recent-search observations do not prove historical availability before retrieval",
    }
    append_jsonl(RUN_LOG, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    args = parser.parse_args()
    now = now_utc()
    end = datetime.fromisoformat(args.end_time.replace("Z", "+00:00")) if args.end_time else now
    start = datetime.fromisoformat(args.start_time.replace("Z", "+00:00")) if args.start_time else end - timedelta(days=1)
    result = collect(args.query, start, end, args.max_results)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
