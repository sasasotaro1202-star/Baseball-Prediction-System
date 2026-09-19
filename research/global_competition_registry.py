"""Validate the machine-readable global competition registry fail-closed."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/"config/global_baseball_competition_registry.json"
ALLOWED_STATUS={"IMPLEMENTED","CANDIDATE","RESEARCH_ONLY","EXCLUDED"}
ALLOWED_PIT={"NOT_ASSESSED","PARTIAL","PASS","FAIL"}
def load_registry()->dict:
    obj=json.loads(REGISTRY.read_text(encoding="utf-8"))
    if obj.get("schema_version")!=1: raise ValueError("unsupported registry schema")
    comps=obj.get("competitions")
    if not isinstance(comps,list) or not comps: raise ValueError("registry has no competitions")
    seen=set()
    for c in comps:
        cid=c.get("competition_id")
        if not cid or cid in seen: raise ValueError(f"duplicate/missing competition_id: {cid!r}")
        seen.add(cid)
        if c.get("implementation_status") not in ALLOWED_STATUS: raise ValueError(f"invalid status: {cid}")
        if c.get("pit_status") not in ALLOWED_PIT: raise ValueError(f"invalid PIT status: {cid}")
        if c.get("production_eligible") is True and (c.get("pit_status")!="PASS" or c.get("implementation_status")!="IMPLEMENTED"):
            raise ValueError(f"production eligibility without required gates: {cid}")
    return obj
if __name__=="__main__": print(f"validated {len(load_registry()['competitions'])} competition records")
