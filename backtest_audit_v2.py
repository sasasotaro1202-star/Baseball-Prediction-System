#!/usr/bin/env python3
"""Deterministic preflight audit for the baseball backtest pipeline."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
errors = []


def need(path, pattern, label):
    p = ROOT / path
    if not p.exists():
        errors.append(f"missing {path}")
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if not re.search(pattern, text, re.S):
        errors.append(f"missing {label} in {path}")
    return text

npbpatch = need("npb_runtime_patch.py", r"def\s+_official_starters_from_npb\b.*?def\s+first_pitchers\b", "strict official starter resolver")
if npbpatch:
    if "if not rows:" not in npbpatch:
        errors.append("NPB schedule empty-state protection missing")
    if "away_starter" not in npbpatch or "home_starter" not in npbpatch:
        errors.append("starter fields missing from NPB runtime patch")
    if "MIN_STARTER_LINE_COVERAGE" not in npbpatch:
        errors.append("starter coverage gate missing")

runtime = need("baseball_backtest_runtime_patch.py", r"def\s+_normalize_npb_pbp\b", "NPB PBP normalizer")
if runtime:
    if "BACKTEST_RUNTIME_HARDENING_V2" not in runtime:
        errors.append("runtime hardening V2 marker missing")
    if "home_starter" not in runtime or "away_starter" not in runtime:
        errors.append("starter metadata propagation missing")

base = need("baseball_backtest.py", r"def\s+_update_pitcher_history\b", "pitcher history updater")
if base:
    pred = base.find("match_features(row)")
    hist = base.find("_update_pitcher_history(row)")
    if pred >= 0 and hist >= 0 and hist < pred:
        errors.append("pitcher history update occurs before prediction features")

workflow = need(".github/workflows/baseball_backtest.yml", r"timeout-minutes:\s*30", "30-minute job limit")
if workflow:
    for token in ("BASEBALL_TIME_BUDGET_SEC", "data/checkpoints/npb_collection_status.json", "contents: write"):
        if token not in workflow:
            errors.append(f"workflow invariant missing: {token}")
need(".github/workflows/baseball_audit.yml", r"backtest_audit\.py", "audit workflow")

try:
    import pandas as pd
    candidates = [ROOT / "data" / "npb_multi_source_games_all.csv", ROOT / "data" / "npb_aggregate.csv"]
    existing = next((p for p in candidates if p.exists()), None)
    if existing is not None:
        df = pd.read_csv(existing)
        if "game_id" in df.columns and df["game_id"].duplicated().any():
            errors.append("duplicate game_id values in aggregate data")
        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            if dt.isna().all():
                errors.append("no valid timestamps in aggregate datetime")
except Exception as exc:
    errors.append(f"aggregate audit failed: {exc}")

if errors:
    print("[AUDIT] FAIL")
    for e in errors:
        print(f"- {e}")
    sys.exit(1)
print("[AUDIT] PASS: code and available-data invariants are valid")
