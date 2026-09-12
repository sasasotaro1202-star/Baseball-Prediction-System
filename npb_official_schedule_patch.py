#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add an authoritative NPB.jp schedule/result validation layer.

The granular SPAIA feed remains the efficient source for per-game enrichment.
When an NPB.jp monthly schedule-detail archive exists, completed rows are
validated against its date, home/away identity and published score. Future
rows are validated by schedule identity only. For historical seasons where
that NPB.jp monthly archive is not available, the configured SPAIA fallback is
retained and explicitly logged; the collector never invents data or silently
swaps home/away teams.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

P = Path("npb_multi_source.py")
s = P.read_text(encoding="utf-8")
# Make the dependency permanent in the collector so an already-patched or
# partially restored source cannot raise NameError when pd.read_html receives
# an in-memory HTML buffer.
head = s.split("def ", 1)[0]
if "import io" not in head:
    if "import os, re, time, json, html\n" in s:
        s = s.replace("import os, re, time, json, html\n", "import os, re, time, json, html, io\n", 1)
    elif "import os, re, time, json\n" in s:
        s = s.replace("import os, re, time, json\n", "import os, re, time, json, io\n", 1)
    else:
        s = s.replace("from pathlib import Path\n", "from pathlib import Path\nimport io\n", 1)

MARKER = "# OFFICIAL_NPB_SCHEDULE_VALIDATION_V1"
if MARKER in s:
    P.write_text(s, encoding="utf-8")
    print("[OFFICIAL PATCH] V1 already applied; import io verified")
    raise SystemExit(0)

anchor = "def fetch_games(year):\n"
if anchor not in s:
    raise RuntimeError("fetch_games anchor not found")

insert = r'''
# OFFICIAL_NPB_SCHEDULE_VALIDATION_V1
_OFFICIAL_SCHEDULE_CACHE = {}


def _official_schedule_rows(year):
    """Return canonical (date, home, away, home_score, away_score, score_known)."""
    if year in _OFFICIAL_SCHEDULE_CACHE:
        return _OFFICIAL_SCHEDULE_CACHE[year]
    rows = []
    complete_fetch = True
    for month in range(3, 12):
        if near_deadline():
            complete_fetch = False
            break
        url = f"{NPB}/games/{year}/schedule_{month:02d}_detail.html"
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0 baseball-backtest"})
            r.raise_for_status()
            tables = pd.read_html(io.StringIO(r.text))
        except Exception as exc:
            complete_fetch = False
            print(f"[OFFICIAL SCHEDULE SKIP] year={year} month={month}: {exc}")
            break
        found_table = False
        for table in tables:
            if "月日" not in table.columns or "対戦カード" not in table.columns:
                continue
            found_table = True
            for _, rec in table.iterrows():
                date_text = str(rec.get("月日", ""))
                m = re.search(r"(\d{1,2})/(\d{1,2})", date_text)
                if not m:
                    continue
                mm, dd = int(m.group(1)), int(m.group(2))
                item = re.sub(r"\s+", " ", str(rec.get("対戦カード", ""))).strip()
                item = re.sub(r"\([^)]*\)", "", item).strip()
                if not item or item in ("-", "nan"):
                    continue
                score = re.match(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$", item)
                if score:
                    home_raw, home_score, away_score, away_raw = score.groups()
                    score_known = True
                else:
                    matchup = re.match(r"^(.+?)\s*-\s*(.+?)$", item)
                    if not matchup:
                        continue
                    home_raw, away_raw = matchup.groups()
                    home_score = away_score = None
                    score_known = False
                home = official_name(home_raw.strip())
                away = official_name(away_raw.strip())
                if not home or not away or home == away:
                    continue
                rows.append((f"{year:04d}-{mm:02d}-{dd:02d}", home, away, int(home_score) if score_known else None, int(away_score) if score_known else None, score_known))
        if not found_table:
            complete_fetch = False
            print(f"[OFFICIAL SCHEDULE SKIP] year={year} month={month}: expected NPB.jp schedule table missing")
            break
    if not complete_fetch:
        rows = []
    _OFFICIAL_SCHEDULE_CACHE[year] = rows
    return rows


def _official_validate_games(year, games):
    """Validate SPAIA rows against NPB.jp; use SPAIA fallback when archive is absent."""
    if games.empty:
        return games
    official = _official_schedule_rows(year)
    if not official:
        print(f"[OFFICIAL SCHEDULE FALLBACK] year={year} NPB.jp monthly archive unavailable; retaining SPAIA rows")
        return games
    index = {(d, home, away): (hs, aas, known) for d, home, away, hs, aas, known in official}
    kept = []
    rejected = 0
    score_checked = 0
    for r in games.itertuples(index=False):
        d = pd.Timestamp(r.datetime).strftime("%Y-%m-%d")
        got = index.get((d, str(r.home), str(r.away)))
        if got is None:
            rejected += 1
            continue
        official_home_score, official_away_score, score_known = got
        if score_known:
            score_checked += 1
            if int(round(float(r.home_score))) != official_home_score or int(round(float(r.away_score))) != official_away_score:
                raise RuntimeError(f"official NPB result mismatch: {r.game_id} {d} {r.home}-{r.away} SPAIA={r.home_score}-{r.away_score} NPB={official_home_score}-{official_away_score}")
        kept.append(r._asdict())
    if rejected:
        print(f"[OFFICIAL SCHEDULE] year={year} rejected_unverified_rows={rejected}")
    print(f"[OFFICIAL SCHEDULE] year={year} admitted={len(kept)} score_checked={score_checked}")
    if not kept:
        raise RuntimeError(f"official NPB schedule validation admitted zero games for {year}")
    return pd.DataFrame(kept, columns=games.columns).sort_values(["datetime", "game_id"]).reset_index(drop=True)


'''
s = s.replace(anchor, insert + anchor, 1)
s = s.replace("def fetch_games(year):\n", "def _fetch_games_spaia(year):\n", 1)
wrapper = "\ndef fetch_games(year):\n    games = _fetch_games_spaia(year)\n    return _official_validate_games(year, games)\n"
marker2 = "\ndef load_checkpoint(year):\n"
if marker2 not in s:
    raise RuntimeError("load_checkpoint anchor not found")
s = s.replace(marker2, wrapper + marker2, 1)
P.write_text(s, encoding="utf-8")
print("[OFFICIAL PATCH] V1 applied: NPB.jp schedule identity + published result validation enabled")
