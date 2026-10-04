"""Strict verifier for the extreme representation research artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from research.extreme_representation_lab import (
    REPRESENTATIONS, SEED_FAMILY_PATTERNS, HALF_LIVES, MODEL_POOLS, MODEL_PROFILES,
)

def validate(path: str) -> dict:
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("research_contract_id") != "extreme-representation-v1": raise RuntimeError("extreme representation research contract mismatch")
    if obj.get("status") != "RESEARCH_ONLY": raise RuntimeError("artifact is not research-only")
    if obj.get("decision") != "NO_AUTO_ADOPTION": raise RuntimeError("artifact permits auto-adoption")
    sha=str(obj.get("git_commit_sha",""))
    if len(sha)!=40 or any(ch not in "0123456789abcdef" for ch in sha.lower()): raise RuntimeError("artifact Git snapshot SHA is missing or invalid")
    counts=obj.get("stage_counts",{})
    expected_a=len(REPRESENTATIONS)*len(SEED_FAMILY_PATTERNS)
    expected_b=12*len(HALF_LIVES)*len(MODEL_POOLS)
    expected_c=4*len(MODEL_PROFILES)
    if counts.get("stage_a_requested")!=expected_a: raise RuntimeError("Stage A breadth mismatch")
    if counts.get("stage_b_requested")!=expected_b: raise RuntimeError("Stage B breadth mismatch")
    if counts.get("stage_c_requested")!=expected_c: raise RuntimeError("Stage C breadth mismatch")
    if int(counts.get("stage_a_successful",0))+int(counts.get("stage_a_failed",0))!=expected_a: raise RuntimeError("Stage A execution does not reconcile")
    if int(counts.get("stage_b_successful",0))+int(counts.get("stage_b_failed",0))!=expected_b: raise RuntimeError("Stage B execution does not reconcile")
    if int(counts.get("stage_c_successful",0))+int(counts.get("stage_c_failed",0))!=expected_c: raise RuntimeError("Stage C execution does not reconcile")
    gate=obj.get("research_gate",{})
    if gate.get("holdout_locked_before_selection") is not True: raise RuntimeError("holdout lock missing")
    if gate.get("no_auto_adoption") is not True: raise RuntimeError("no-auto-adoption missing")
    if int(gate.get("unexpected_execution_failures",0))!=0: raise RuntimeError("unexpected execution failures exist")
    catalog=obj.get("representation_catalog",{})
    if catalog.get("modes")!=list(REPRESENTATIONS): raise RuntimeError("representation catalog mismatch")
    if catalog.get("model_profiles")!=list(MODEL_PROFILES): raise RuntimeError("model profile catalog mismatch")
    holdout=obj.get("locked_holdout",{})
    if holdout.get("winner_only") is not True: raise RuntimeError("holdout must be winner-only")
    if int(obj.get("locked_holdout_rows",0))<1: raise RuntimeError("holdout empty")
    for k in ("LogLoss","Brier","Accuracy","ECE","rows"):
        if k not in holdout.get("metrics",{}): raise RuntimeError(f"holdout metric missing: {k}")
    winner=obj.get("winner",{})
    if not str(winner.get("candidate_id","")).strip(): raise RuntimeError("winner missing")
    return {"league":obj.get("league"),"status":obj.get("status"),"stage_counts":counts,"winner":winner["candidate_id"],"holdout_rows":obj["locked_holdout_rows"]}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("path"); a=p.parse_args()
    print(json.dumps(validate(a.path),ensure_ascii=False,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
