#!/usr/bin/env python3
"""Exact-parallel Baseball verification worker.

The canonical Baseball engine and feature generation remain unchanged. Each
worker evaluates a disjoint subset of the existing chronological retraining
blocks; every block still trains on the complete prior prefix. This preserves
OOS semantics while distributing the expensive model-fit work.
"""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,shutil
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
W=int(os.environ["WORKER_ID"]); N=int(os.environ.get("WORKER_COUNT","4"))
OUT=ROOT/"exact_parallel"/"out"; OUT.mkdir(parents=True,exist_ok=True)

patches=["repair_npb_syntax.py","npb_runtime_patch.py","npb_official_schedule_patch.py","npb_quality_runtime_patch.py","baseball_backtest_runtime_patch.py","baseball_production_runtime_patch.py","baseball_mlb_score_hilo_patch.py","baseball_quality_runtime_patch.py"]
for patch in patches:
    subprocess.run([sys.executable,patch],cwd=ROOT,check=True)
subprocess.run([sys.executable,"exact_parallel/shard_patch.py"],cwd=ROOT,check=True)

# Never resume a prior worker checkpoint. The frozen data snapshot is the sole
# input and the canonical feature builder reconstructs the full past-only X.
shutil.rmtree(ROOT/"results"/"checkpoints",ignore_errors=True)
for p in (ROOT/"results"/"npb_backtest_results.csv",ROOT/"results"/"mlb_backtest_results.csv",ROOT/"results"/"combined_backtest_results.csv"):
    p.unlink(missing_ok=True)

env=os.environ.copy(); env.update({"BASEBALL_TIME_BUDGET_SEC":"3600","MLB_ENRICH_STARTERS":"0","BASEBALL_EXACT_SHARD_ID":str(W),"BASEBALL_EXACT_SHARD_COUNT":str(N)})
subprocess.run([sys.executable,"baseball_backtest.py","--npb-only","--data-dir","data"],cwd=ROOT,env=env,check=True)
subprocess.run([sys.executable,"baseball_backtest.py","--mlb-only","--data-dir","data","--mlb-start","2020","--mlb-end","2026"],cwd=ROOT,env=env,check=True)

files=[("NPB","results/npb_backtest_results.csv"),("MLB","results/mlb_backtest_results.csv")]
manifest={"worker":W,"count":N,"code_sha":os.environ.get("GITHUB_SHA","")}
for league,rel in files:
    p=ROOT/rel
    if not p.exists(): raise SystemExit(f"missing {rel}")
    d=pd.read_csv(p,low_memory=False)
    if d.empty: raise SystemExit(f"empty {league} worker output")
    # game_id is the stable identity; block sharding produces disjoint rows.
    if "game_id" not in d.columns: raise SystemExit(f"{league}: game_id missing")
    s=d.copy(); s.to_csv(OUT/f"{league.lower()}_worker_{W}.csv",index=False)
    manifest[league]={"rows_shard":len(s),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(OUT/f"baseball_worker_{W}.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(manifest,ensure_ascii=False))
