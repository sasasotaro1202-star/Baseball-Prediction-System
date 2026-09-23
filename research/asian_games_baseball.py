"""Fail-closed Aichi-Nagoya 2026 Asian Games baseball schedule/research adapter.

Only official organizer/BFJ sources are accepted. This module collects schedule
evidence and provenance for the active research lane but does not manufacture
starter timing, odds, or prediction eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OFFICIAL_NEWS = "https://www.aichi-nagoya2026.org/ja/news-2050/"
OFFICIAL_BASEBALL = "https://www.aichi-nagoya2026.org/ja/sport/baseball/"
OFFICIAL_BFJ_EVENTS = "https://www.baseballjapan.org/jpn/system/prog/event_index.php?domain=&lang=&year=2026"
OUT = Path("results/asian_games_baseball_schedule.json")

_TIMEOUT = 20
_ATTEMPTS = 4
_RETRYABLE = {408, 429, 500, 502, 503, 504}

def _fetch_text(url: str, timeout: int = _TIMEOUT, attempts: int = _ATTEMPTS) -> tuple[str, str]:
    """Fetch official text with bounded retry; final failure remains visible."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Baseball-Prediction-System/1.0",
                         "Accept": "text/html,application/xhtml+xml"},
            )
            if r.status_code in _RETRYABLE and attempt < attempts:
                time.sleep(min(2.0 * attempt, 8.0))
                continue
            r.raise_for_status()
            return r.text, datetime.now(timezone.utc).isoformat()
        except requests.RequestException as exc:
            last = exc
            if attempt < attempts:
                time.sleep(min(2.0 * attempt, 8.0))
            else:
                raise RuntimeError(f"official source fetch failed after {attempts} attempts: {url}") from exc
    raise RuntimeError(f"official source fetch failed: {url}") from last

def _extract_bbl_codes(body: str) -> list[str]:
    """Extract official baseball session codes deterministically."""
    codes = sorted(set(re.findall(r"\bBBL\d{2}\b", body, flags=re.I)))
    return [x.upper() for x in codes]

def _extract_schedule_rows(body: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        tables = pd.read_html(body)
    except (ValueError, ImportError):
        tables = []
    for table in tables:
        text = table.to_string(index=False)
        if "BBL" not in text:
            continue
        for _, row in table.iterrows():
            values = [str(x).strip() for x in row.tolist()]
            if any(re.fullmatch(r"BBL\d{2}", x, flags=re.I) for x in values):
                rows.append({"raw": values})
    return rows

def _validate_bfj_event_page(body: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", body)
    has_asian_games = "アジア競技大会" in normalized or "Asian Games" in normalized
    has_baseball = "野球" in normalized or "Baseball" in normalized
    return {"status": "PASS" if has_asian_games and has_baseball else "BLOCKED",
            "asian_games_marker": bool(has_asian_games),
            "baseball_marker": bool(has_baseball)}

def fetch_schedule(url: str = OFFICIAL_NEWS, *, baseball_url: str = OFFICIAL_BASEBALL,
                   bfj_url: str = OFFICIAL_BFJ_EVENTS, timeout: int = _TIMEOUT,
                   output: str | Path = OUT) -> dict[str, Any]:
    """Collect official schedule evidence from organizer + BFJ registry."""
    news_body, news_retrieved = _fetch_text(url, timeout=timeout)
    baseball_body, baseball_retrieved = _fetch_text(baseball_url, timeout=timeout)
    bfj_body, bfj_retrieved = _fetch_text(bfj_url, timeout=timeout)
    schedule_rows = _extract_schedule_rows(news_body)
    codes = sorted(set(_extract_bbl_codes(news_body) + _extract_bbl_codes(baseball_body)))
    bfj_check = _validate_bfj_event_page(bfj_body)
    if bfj_check["status"] != "PASS":
        raise RuntimeError("BFJ official 2026 event registry lacks expected Asian Games baseball markers.")
    if not codes:
        raise RuntimeError("Official Asian Games baseball sources yielded no BBL session codes.")
    payload = {
        "schema_version": "asian-games-baseball-research-v2",
        "competition_id": "asian_games_baseball",
        "status": "RESEARCH_ONLY",
        "source_priority": ["BFJ_EVENT_REGISTRY", "ORGANIZER_SCHEDULE_UPDATE", "ORGANIZER_BASEBALL_PAGE"],
        "sources": {
            "bfj_event_registry": {"url": bfj_url, "retrieved_at": bfj_retrieved,
                                   "sha256": hashlib.sha256(bfj_body.encode("utf-8")).hexdigest(),
                                   "validation": bfj_check},
            "organizer_schedule_update": {"url": url, "retrieved_at": news_retrieved,
                                          "sha256": hashlib.sha256(news_body.encode("utf-8")).hexdigest()},
            "organizer_baseball_page": {"url": baseball_url, "retrieved_at": baseball_retrieved,
                                        "sha256": hashlib.sha256(baseball_body.encode("utf-8")).hexdigest()},
        },
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "bbl_session_codes": codes,
        "schedule_rows": schedule_rows,
        "starter_evidence_status": "NOT_ASSESSED",
        "historical_oos_status": "NOT_READY",
        "prediction_eligible": False,
        "eligibility_reason": ("Starter PIT announcement evidence, competition-specific historical OOS, "
                               "calibration, and locked holdout are not yet independently validated."),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    payload = fetch_schedule(output=args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())