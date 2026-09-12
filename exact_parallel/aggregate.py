#!/usr/bin/env python3
"""Strict equality gate for four exact-parallel Baseball workers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
for league in ("npb","mlb"):
    base=ROOT/"exact_parallel"/"baseline"/f"{league}_backtest_results.csv"
    workers=[ROOT/"exact_parallel"/"out"/f"{league}_worker_{i}.csv" for i in range(4)]
    if not base.exists(): raise SystemExit(f"missing baseline {league}")
    if not all(p.exists() for p in workers): raise SystemExit(f"missing worker {league}")
    b=pd.read_csv(base,low_memory=False); a=pd.concat([pd.read_csv(p,low_memory=False) for p in workers],ignore_index=True)
    if "game_id" not in b.columns or "game_id" not in a.columns: raise SystemExit(f"{league}: game_id missing")
    if len(a)!=len(b): raise SystemExit(f"{league}: row count mismatch baseline={len(b)} merged={len(a)}")
    if a.game_id.astype(str).duplicated().any(): raise SystemExit(f"{league}: duplicate game_id across workers")
    if set(a.game_id.astype(str))!=set(b.game_id.astype(str)): raise SystemExit(f"{league}: game_id gap/overlap")
    key=b.game_id.astype(str).tolist(); a=a.assign(__key=a.game_id.astype(str)).set_index("__key").loc[key].reset_index(drop=True)
    if list(a.columns)!=list(b.columns): raise SystemExit(f"{league}: schema mismatch")
    for c in b.columns:
        if pd.api.types.is_numeric_dtype(b[c]):
            x=a[c].to_numpy(float); y=b[c].to_numpy(float)
            if not np.array_equal(x,y,equal_nan=True):
                d=np.nanmax(np.abs(x-y)); raise SystemExit(f"{league}: numeric mismatch {c} max_abs={d}")
        elif not a[c].astype(object).equals(b[c].astype(object)):
            raise SystemExit(f"{league}: column mismatch {c}")
    a.to_csv(ROOT/"exact_parallel"/f"merged_{league}_backtest_results.csv",index=False)
(Path(ROOT/"exact_parallel"/"equality_gate.json")).write_text(json.dumps({"status":"PASS","workers":4,"message":"NPB and MLB four-worker outputs are exactly identical to the single-run baseline."},ensure_ascii=False,indent=2),encoding="utf-8")
print(Path(ROOT/"exact_parallel"/"equality_gate.json").read_text())
