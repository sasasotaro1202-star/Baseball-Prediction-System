"""Fail-closed audit for the Future Generalization v6 research artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED = ("schema_version", "git_commit", "dataset_hash", "pit", "development_oos", "holdout", "controller")
NUMERIC_METRICS = ("Accuracy", "LogLoss", "Brier")


def _check(ok: bool, reason: str) -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", "reason": reason}


def _finite_metric_map(obj: object, keys: tuple[str, ...]) -> bool:
    if not isinstance(obj, dict):
        return False
    try:
        return all(k in obj and float(obj[k]) == float(obj[k]) for k in keys)
    except Exception:
        return False


def audit_artifact(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"artifact_missing:{p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"artifact_unreadable:{type(exc).__name__}") from exc

    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError("artifact_schema_missing:" + ",".join(missing))

    pit = data["pit"]
    dev = data["development_oos"]
    hold = data["holdout"]
    ctl = data["controller"]
    baseline = hold.get("baseline")
    new = hold.get("v6_full", hold.get("full_architecture"))

    checks = {
        "PIT": _check(
            pit.get("status") == "PASS"
            and pit.get("future_holdout_excluded_from_fit") is True,
            "PIT gate passed and final holdout was excluded from fitting/tuning",
        ),
        "Leakage": _check(
            ctl.get("production_changed") is False
            and int(hold.get("rows", 0)) > 0
            and int(dev.get("rows", 0)) > int(hold.get("rows", 0)),
            "chronological unseen holdout exists and Production remained isolated",
        ),
        "Meta-Leakage": _check(
            int(dev.get("rows", 0)) > 0
            and int(dev.get("validation_windows", 0)) >= 2
            and int(dev.get("forward_meta_cut_rows", 0)) > 0
            and int(dev.get("forward_meta_eval_rows", 0)) > 0
            and int(dev.get("forward_meta_cut_rows", 0)) + int(dev.get("forward_meta_eval_rows", 0))
            == int(dev.get("rows", 0)),
            "meta layers use an earlier Development-OOS prefix and later Development-OOS evaluation suffix",
        ),
        "Nested OOS": _check(
            bool(dev.get("nested_oos", False)) is True,
            "base OOS precedes meta fitting, meta evaluation, then frozen final holdout",
        ),
        "Calibration / Risk Control": _check(
            isinstance(hold.get("conformal"), dict)
            and isinstance(hold.get("selective"), dict),
            "conformal and selective-risk artifacts are present",
        ),
        "Robustness": _check(
            isinstance(hold.get("stress"), dict)
            and _finite_metric_map(hold.get("stress"), ("mean_l1_change", "p95_l1_change", "unstable_rate")),
            "distribution stress metrics are present and finite",
        ),
        "Artifact Integrity": _check(
            bool(data.get("dataset_hash"))
            and _finite_metric_map(baseline, ("Accuracy", "LogLoss", "Brier"))
            and _finite_metric_map(new, ("Accuracy", "LogLoss", "Brier"))
            and isinstance(ctl.get("fallback"), list)
            and isinstance(hold.get("safety"), dict),
            "dataset hash, metrics, safety and fallback payloads are present",
        ),
        "Promotion": _check(
            ctl.get("promotion") == "HOLD",
            "research artifact cannot self-promote",
        ),
    }

    delta = {
        "Accuracy": float(new["Accuracy"] - baseline["Accuracy"]),
        "LogLoss": float(new["LogLoss"] - baseline["LogLoss"]),
        "Brier": float(new["Brier"] - baseline["Brier"]),
    }
    checks["Metric Completeness"] = _check(
        _finite_metric_map(baseline, NUMERIC_METRICS) and _finite_metric_map(new, NUMERIC_METRICS),
        "baseline and v6 metrics contain finite Accuracy/LogLoss/Brier values",
    )
    passed = all(x["status"] == "PASS" for x in checks.values())
    return {
        "schema_version": "future-generalization-v6-audit-v2",
        "artifact": str(p),
        "checks": checks,
        "metrics": {"baseline": baseline, "new": new, "delta": delta},
        "selective": hold.get("selective"),
        "conformal": hold.get("conformal"),
        "stress": hold.get("stress"),
        "mean_predictability": hold.get("mean_predictability"),
        "mean_confidence": hold.get("mean_confidence"),
        "abstain_rate": hold.get("abstain_rate"),
        "statistical_validation": hold.get("statistical_validation"),
        "overall": "PASS" if passed else "FAIL",
        "promotion": "HOLD",
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Maximum Future-Generalization v6 Audit",
        "",
        f"Overall: **{report.get('overall', 'FAIL')}**",
        f"Promotion: **{report.get('promotion', 'HOLD')}**",
        "",
        "| Gate | Status | Reason |",
        "|---|---|---|",
    ]
    for gate, item in report.get("checks", {}).items():
        lines.append(f"| {gate} | {item['status']} | {item['reason']} |")
    d = report.get("metrics", {}).get("delta", {})
    lines += [
        "",
        "## Metrics",
        f"- Accuracy Δ: {d.get('Accuracy')}",
        f"- LogLoss Δ: {d.get('LogLoss')}",
        f"- Brier Δ: {d.get('Brier')}",
        f"- Abstain rate: {report.get('abstain_rate')}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python research/future_generalization_v6_audit.py ARTIFACT.json", file=sys.stderr)
        return 2
    artifact = Path(sys.argv[1])
    out = artifact.with_name("future_generalization_v6_audit.json")
    md = artifact.with_name("future_generalization_v6_report.md")
    try:
        report = audit_artifact(artifact)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(report, md)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["overall"] == "PASS" else 1
    except Exception as exc:
        report = {
            "schema_version": "future-generalization-v6-audit-v2",
            "overall": "FAIL",
            "promotion": "HOLD",
            "error": f"{type(exc).__name__}: {exc}",
        }
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(report, md)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
