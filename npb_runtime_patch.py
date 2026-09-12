#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust NPB runtime hardening patch.

The collector has evolved since the original patch was written.  This patch
uses function-boundary replacement instead of assuming helper ordering, so a
minor upstream refactor cannot make CI fail before the actual collector runs.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

P = Path("npb_multi_source.py")
s = P.read_text(encoding="utf-8")
MARKER = "# RUNTIME_HARDENING_V5"

if MARKER in s:
    print("[RUNTIME PATCH] V5 already applied")
    raise SystemExit(0)

# Keep imports self-contained.  The official NPB starter fallback is used only
# when the SPAIA historical first-pitcher evidence cannot resolve both sides.
if "import html" not in s.split("def ", 1)[0]:
    s = s.replace("import os, re, time, json\n", "import os, re, time, json, html\n", 1)

start = s.find("def first_pitchers(")
if start < 0:
    raise RuntimeError("first_pitchers function not found")
end = s.find("\ndef _walk_dicts", start)
if end < 0:
    raise RuntimeError("first_pitchers function boundary not found")

new_func = r'''def _official_starters_from_npb(game_id, home="", away=""):
    """Resolve historical starters from the official NPB.jp play-by-play page."""
    team_codes = {
        "読売ジャイアンツ":"g", "東京ヤクルトスワローズ":"s", "横浜DeNAベイスターズ":"db",
        "広島東洋カープ":"c", "阪神タイガース":"t", "中日ドラゴンズ":"d",
        "福岡ソフトバンクホークス":"h", "埼玉西武ライオンズ":"l", "北海道日本ハムファイターズ":"f",
        "千葉ロッテマリーンズ":"m", "東北楽天ゴールデンイーグルス":"e", "オリックス・バファローズ":"bs",
    }
    gid = re.sub(r"\.0$", "", str(game_id).strip())
    if len(gid) < 10 or not gid[:8].isdigit():
        return "", ""
    year, md, number = gid[:4], gid[4:8], gid[-2:]
    hc = team_codes.get(official_name(home))
    ac = team_codes.get(official_name(away))
    if not hc or not ac:
        return "", ""
    url = f"{NPB}/scores/{year}/{md}/{ac}-{hc}-{number}/playbyplay.html"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0 baseball-backtest"})
        if r.status_code != 200:
            return "", ""
        text = html.unescape(r.text)
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        # NPB.jp labels starting pitchers explicitly on the play-by-play page.
        names = re.findall(r"(?:\(先発投手\)|（先発投手）)\s*([^<|\s]+)", text)
        if len(names) >= 2:
            return names[0].strip(), names[1].strip()
        names = re.findall(r"先発投手\)?[）)]?\s*[:：]?\s*([^<|\s]+)", text)
        if len(names) >= 2:
            return names[0].strip(), names[1].strip()
    except Exception as exc:
        print(f"[OFFICIAL STARTER FALLBACK SKIP] game={game_id}: {exc}")
    return "", ""


def first_pitchers(game_id, home="", away=""):
    try:
        raw = get_json(f'{SPAIA}/flash_atbat_history', {'gameId': game_id})
    except Exception:
        raw = None
    if isinstance(raw, list):
        arr = []
        for x in raw:
            if not isinstance(x, dict):
                continue
            pid = x.get('pitId', x.get('pitcher', x.get('PitcherCD', '')))
            ser = str(x.get('fiveDigitSerialNumber', ''))
            if pid in (None, '', 0) or not ser:
                continue
            arr.append((ser, str(pid)))
        arr.sort(key=lambda z: z[0])
        away_id = home_id = ''
        seen = set()
        for ser, pid in arr:
            if len(ser) < 3:
                continue
            half = ser[2]
            inning = ser[:2]
            key = (inning, half)
            if key in seen:
                continue
            seen.add(key)
            if half in ('T', 't', '1') and not away_id:
                away_id = pid
            elif half in ('B', 'b', '2') and not home_id:
                home_id = pid
            if away_id and home_id:
                break
        if away_id and home_id:
            return away_id, home_id
    # Strict rule: unresolved starters remain unresolved; never pick arbitrary pitchers.
    official_away, official_home = _official_starters_from_npb(game_id, home, away)
    if official_away and official_home:
        return official_away, official_home
    return "", ""
'''
s = s[:start] + new_func + s[end:]

# Pass team identity into the resolver so the official NPB fallback can build
# the canonical play-by-play URL.  Use a targeted replacement that matches the
# current compact collector as well as formatted variants.
old = "away,home=first_pitchers(r.game_id);date="
if old in s:
    s = s.replace(old, "away,home=first_pitchers(r.game_id,r.home,r.away);date=", 1)
else:
    pat = re.compile(r"away,home=first_pitchers\(r\.game_id\)\s*;\s*date=")
    s, n = pat.subn("away,home=first_pitchers(r.game_id,r.home,r.away);date=", s, count=1)
    if n != 1:
        raise RuntimeError("enrich_one first_pitchers call not found")

# Keep the patch auditable.  These fields do not affect model features; they
# only document the intended provenance of prediction-critical enrichment.
prov = "z['home_starter_line_ok']=bool(hm);z['away_starter_line_ok']=bool(am);z['_player_rows']=players"
if prov in s:
    s = s.replace(
        prov,
        "z['home_starter_line_ok']=bool(hm);z['away_starter_line_ok']=bool(am);z['starter_source']='SPAIA game API + NPB.jp official fallback';z['player_source']='SPAIA game API';z['weather_source']='Open-Meteo archive';z['_player_rows']=players",
        1,
    )

s = MARKER + "\n" + s
P.write_text(s, encoding="utf-8")
print("[RUNTIME PATCH] V5 applied: robust function replacement + strict official NPB starter fallback + provenance")
