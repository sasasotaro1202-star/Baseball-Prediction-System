#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent production hardening for the baseball backtest."""
from __future__ import annotations
from pathlib import Path
import runpy
import re

P = Path("baseball_backtest.py")
s = P.read_text(encoding="utf-8")

# Runtime budget: tolerate every prior patch form and also the unpatched base
# form. Never fail merely because another patch changed the surrounding text.
if "# BASEBALL_RUNTIME_BUDGET_HARDENING_V3" not in s:
    lines = s.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if "self.time_budget_sec" in line and "=" in line:
            indent = line[:len(line)-len(line.lstrip())]
            lines[i] = indent + 'self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 3600.0)  # production ceiling: 60:00'
            replaced = True
            break
    if not replaced:
        # Future-safe fallback: inject the budget immediately after started_at
        # inside __init__, which is the stable lifecycle point used by the engine.
        needle = "        self.started_at = time.time()"
        if needle not in s:
            raise SystemExit("[PRODUCTION PATCH] could not locate backtest timing initialization")
        s = s.replace(needle, needle + '\n        self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 3600.0)  # production ceiling: 60:00', 1)
    else:
        s = "\n".join(lines) + ("\n" if s.endswith("\n") else "")
    s = re.sub(r"# BASEBALL_RUNTIME_BUDGET_HARDENING_V\d+\n", "", s)
    s = "# BASEBALL_RUNTIME_BUDGET_HARDENING_V3\n" + s
    s = s.replace("210-minute production limit", "60-minute production limit")
    s = s.replace("1500.0)  # hard cap: 29:00", "3600.0)  # production ceiling: 60:00")
    P.write_text(s, encoding="utf-8")
    print("[PRODUCTION PATCH] runtime budget hardened")
else:
    print("[PRODUCTION PATCH] runtime budget already hardened")

s = P.read_text(encoding="utf-8")
# Starter gate: insert before the first feature build inside run_walkforward.
if "# BASEBALL_PRODUCTION_HARDENING_V2" not in s:
    target = "        X, y, meta = self.build_features(games)"
    if target not in s:
        raise SystemExit("[PRODUCTION PATCH] feature-build insertion point missing")
    gate = '''        # STRICT STARTER COVERAGE GATE
        # Historical OOS evaluation must represent a valid pregame state.
        # NPB requires >=70%; MLB requires >=90% both-starter coverage.
        if "home_starter" in games.columns and "away_starter" in games.columns:
            hs = games["home_starter"].fillna("").astype(str).str.strip().str.len() > 0
            aw = games["away_starter"].fillna("").astype(str).str.strip().str.len() > 0
            starter_rate = float((hs & aw).mean()) if len(games) else 0.0
            threshold = 0.90 if league == "MLB" else 0.70
            self.audit.append({"type": f"{league.lower()}_starter_coverage", "games": int(len(games)), "both_starter_rate": starter_rate, "required_rate": threshold})
            print(f"[{league} AUDIT] both-starter coverage={starter_rate:.1%} required={threshold:.0%}")
            if starter_rate < threshold:
                raise RuntimeError(f"{league} starter coverage too low: {starter_rate:.1%}; required >= {threshold:.0%}")

'''
    s = s.replace(target, gate + target, 1)
    s = re.sub(r"# BASEBALL_PRODUCTION_HARDENING_V\d+\n", "", s)
    s = "# BASEBALL_PRODUCTION_HARDENING_V2\n" + s
    P.write_text(s, encoding="utf-8")
    print("[PRODUCTION PATCH] starter gate hardened")
else:
    print("[PRODUCTION PATCH] starter gate already hardened")

# Apply idempotent source/data/model hardening chain. Each patch must be present.
for patch in (
    "npb_official_schedule_patch.py",
    "npb_quality_runtime_patch.py",
    "baseball_mlb_score_hilo_patch.py",
    "baseball_quality_runtime_patch.py",
):
    p = Path(patch)
    if not p.exists():
        raise SystemExit(f"[PRODUCTION PATCH] required patch missing: {patch}")
    runpy.run_path(str(p), run_name="__main__")

print("[PRODUCTION PATCH] complete")
