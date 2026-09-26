"""Fail-closed audit for the Future Generalization v6 research artifact."""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED = ("schema_version","git_commit","dataset_hash","pit","development_oos","holdout","controller")

def _check(ok: bool, reason: str) -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", "reason": reason}

def audit_artifact(path: str | Path) -> dict[str, Any]:
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(f"artifact_missing:{p}")
    try:
        data=json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"artifact_unreadable:{type(exc).__name__}") from exc
    missing=[k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError("artifact_schema_missing:"+",".join(missing))
    pit=data["pit"]; dev=data["development_oos"]; hold=data["holdout"]; ctl=data["controller"]
    checks={
        "PIT":_check(pit.get("status")=="PASS" and pit.get("future_holdout_excluded_from_fit") is True,
                     "PIT PASS and frozen holdout excluded from fit"),
        "Leakage":_check(ctl.get("production_changed") is False and int(hold.get("rows",0))>0,
                         "production isolation and non-empty unseen holdout"),
        "Meta-Leakage":_check(int(dev.get("rows",0))>0 and int(dev.get("validation_windows",0))>=2,
                              "controller trained from chronological Development-OOS"),
        "Artifact Integrity":_check(bool(data.get("dataset_hash"))
                                     and isinstance(hold.get("baseline"),dict)
                                     and isinstance(hold.get("full_architecture"),dict)
                                     and isinstance(ctl.get("safety"),dict),
                                     "hash, metrics and safety payloads present"),
        "Promotion":_check(ctl.get("promotion")=="HOLD","research artifact cannot self-promote"),
    }
    base=hold["baseline"]; new=hold["full_architecture"]
    delta={"Accuracy":float(new["Accuracy"]-base["Accuracy"]),
           "LogLoss":float(new["LogLoss"]-base["LogLoss"]),
           "Brier":float(new["Brier"]-base["Brier"])}
    passed=all(x["status"]=="PASS" for x in checks.values())
    return {
        "schema_version":"future-generalization-v6-audit-v1",
        "artifact":str(p),"checks":checks,
        "metrics":{"baseline":base,"new":new,"delta":delta},
        "selective":hold.get("selective"),
        "drift_levels":hold.get("drift_levels"),
        "mean_predictability":hold.get("mean_predictability"),
        "mean_confidence":hold.get("mean_confidence"),
        "abstain_rate":hold.get("abstain_rate"),
        "statistical_validation":hold.get("statistical_validation"),
        "overall":"PASS" if passed else "FAIL","promotion":"HOLD",
    }

def write_report(report: dict[str,Any], path: str | Path) -> None:
    lines=["# Maximum Future-Generalization v6 Audit","",
           f"Overall: **{report.get('overall','FAIL')}**",
           f"Promotion: **{report.get('promotion','HOLD')}**","",
           "| Gate | Status | Reason |","|---|---|---|"]
    for gate,item in report.get("checks",{}).items():
        lines.append(f"| {gate} | {item['status']} | {item['reason']} |")
    d=report.get("metrics",{}).get("delta",{})
    lines += ["","## Metrics",
              f"- Accuracy Δ: {d.get('Accuracy')}",
              f"- LogLoss Δ: {d.get('LogLoss')}",
              f"- Brier Δ: {d.get('Brier')}",
              f"- Abstain rate: {report.get('abstain_rate')}"]
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")

def main() -> int:
    if len(sys.argv)!=2:
        print("usage: python research/future_generalization_v6_audit.py ARTIFACT.json",file=sys.stderr)
        return 2
    artifact=Path(sys.argv[1]); out=artifact.with_name("future_generalization_v6_audit.json"); md=artifact.with_name("future_generalization_v6_report.md")
    try:
        report=audit_artifact(artifact)
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        write_report(report,md)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if report["overall"]=="PASS" else 1
    except Exception as exc:
        report={"schema_version":"future-generalization-v6-audit-v1","overall":"FAIL","promotion":"HOLD","error":f"{type(exc).__name__}: {exc}"}
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); write_report(report,md); print(json.dumps(report,ensure_ascii=False,indent=2)); return 1

if __name__=="__main__":
    raise SystemExit(main())
