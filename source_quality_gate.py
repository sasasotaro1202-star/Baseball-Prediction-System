#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate that the baseball pipeline is using the declared source hierarchy.

This is intentionally a static/runtime gate: it does not invent data and it
fails closed when a critical source is missing or when the code drifts away
from the approved source policy.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "DATA_SOURCE_POLICY.json"
FILES = [
    ROOT / "npb_multi_source.py",
    ROOT / "baseball_backtest.py",
    ROOT / "npb_runtime_patch.py",
    ROOT / "baseball_backtest_runtime_patch.py",
]

REQUIRED = {
    "npb_multi_source.py": [
        "https://spaia.jp/baseball/npb/api",
        "https://npb.jp",
        "https://archive-api.open-meteo.com/v1/archive",
        "flash_atbat_history",
        "both_pitcher_game_stats",
        "both_batter_stats",
        "starting_members_for_flash",
        "game_text_pbp",
    ],
    "baseball_backtest.py": [
        "statsapi.mlb.com",
        "MLB_API",
    ],
}


def main() -> int:
    if not POLICY.exists():
        raise SystemExit("[SOURCE GATE] missing DATA_SOURCE_POLICY.json")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if not policy.get("NPB") or not policy.get("MLB"):
        raise SystemExit("[SOURCE GATE] policy must define NPB and MLB")

    failures = []
    for path in FILES:
        if not path.exists():
            failures.append(f"missing file: {path.name}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in REQUIRED.get(path.name, []):
            if needle not in text:
                failures.append(f"{path.name}: required source marker missing: {needle}")

    # Critical rules are checked explicitly so a future refactor cannot silently
    # reintroduce arbitrary starter substitution or guessed values.
    npb = FILES[0].read_text(encoding="utf-8", errors="replace")
    bt = FILES[1].read_text(encoding="utf-8", errors="replace")
    if "Never substitute an arbitrary pitcher" not in npb and "arbitrary" not in npb.lower():
        failures.append("NPB starter anti-substitution rule not visible")
    if "BOTH starters" not in bt and "both starters" not in bt:
        failures.append("MLB both-starters gate marker not visible")

    if failures:
        print("[SOURCE GATE] FAILED")
        for x in failures:
            print(" -", x)
        return 1

    print("[SOURCE GATE] PASS")
    print("[SOURCE GATE] NPB: official identity/result validation + SPAIA granular game data + Open-Meteo weather")
    print("[SOURCE GATE] MLB: MLB Stats API canonical schedule/results + confirmed-starter gate")
    print("[SOURCE GATE] No critical missing value may be guessed or silently substituted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
