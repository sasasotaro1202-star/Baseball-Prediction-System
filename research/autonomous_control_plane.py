"""Bounded autonomous control plane for scheduled Baseball GitHub Actions."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ACTIVE = {"queued", "pending", "waiting", "requested", "in_progress"}
RECOVERABLE = {"cancelled", "timed_out", "startup_failure"}

@dataclass(frozen=True)
class Target:
    workflow: str
    max_age_hours: float
    heavy: bool = False

TARGETS = (
    Target(".github/workflows/baseball-production-runtime-health.yml", 2.0),
    Target(".github/workflows/baseball_governance_autopilot.yml", 8.0),
    Target(".github/workflows/project_source_provenance_audit.yml", 30.0),
    Target(".github/workflows/baseball_24h_research_autopilot.yml", 30.0, True),
    Target(".github/workflows/npb-production.yml", 10.0, True),
    Target(".github/workflows/npb_prediction_experience_archive.yml", 6.0),
    Target(".github/workflows/npb_experience_reconciliation.yml", 30.0),
    Target(".github/workflows/npb_experience_learning.yml", 12.0),
    Target(".github/workflows/baseball_phase1_gate.yml", 30.0),
    Target(".github/workflows/baseball_universal_readiness.yml", 30.0),
    Target(".github/workflows/baseball_candidate_oos.yml", 192.0, True),
)

ROOT = Path(__file__).resolve().parents[1]

def _gh(args: list[str]) -> str:
    result = subprocess.run(["gh", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=45, env=os.environ.copy())
    return result.stdout

def list_runs(repo: str) -> list[dict[str, Any]]:
    payload = json.loads(_gh(["api", f"repos/{repo}/actions/runs?per_page=100&branch=main"]))
    runs = payload.get("workflow_runs", [])
    return [r for r in runs if r.get("head_branch") == "main"]

def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def dispatch_count(runs: list[dict[str, Any]], workflow: str, now: datetime) -> int:
    cutoff = now - timedelta(hours=24)
    return sum(
        1 for r in runs
        if str(r.get("path", "")).lstrip("/") == workflow
        and r.get("event") == "workflow_dispatch"
        and parse_time(r["created_at"]) >= cutoff
    )

def decide(target: Target, runs: list[dict[str, Any]], now: datetime, cap: int = 2) -> dict[str, Any]:
    matching = [r for r in runs if str(r.get("path", "")).lstrip("/") == target.workflow]
    matching.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    active = [r for r in matching if r.get("status") in ACTIVE]
    attempts = dispatch_count(runs, target.workflow, now)
    result: dict[str, Any] = {
        "workflow": target.workflow,
        "max_age_hours": target.max_age_hours,
        "heavy": target.heavy,
        "dispatch_attempts_24h": attempts,
        "decision": "NOOP",
        "reason": "",
        "latest_run_id": None,
        "latest_status": None,
        "latest_conclusion": None,
        "latest_age_minutes": None,
    }
    if active:
        result["reason"] = "active_run"
        result["latest_run_id"] = active[0].get("id")
        result["latest_status"] = active[0].get("status")
        return result
    if not matching:
        if attempts >= cap:
            result["decision"], result["reason"] = "HOLD", "dispatch_cap_reached_without_history"
        else:
            result["decision"], result["reason"] = "DISPATCH", "no_recent_history"
        return result
    latest = matching[0]
    result["latest_run_id"] = latest.get("id")
    result["latest_status"] = latest.get("status")
    result["latest_conclusion"] = latest.get("conclusion")
    age_minutes = max(0.0, (now - parse_time(latest["created_at"])).total_seconds() / 60.0)
    result["latest_age_minutes"] = round(age_minutes, 2)
    if attempts >= cap:
        result["decision"], result["reason"] = "HOLD", "dispatch_cap_reached"
        return result
    conclusion = latest.get("conclusion")
    if conclusion in RECOVERABLE and age_minutes >= 15:
        result["decision"], result["reason"] = "DISPATCH", f"recoverable_terminal_state:{conclusion}"
    elif conclusion == "success" and age_minutes >= target.max_age_hours * 60:
        result["decision"], result["reason"] = "DISPATCH", "stale_success"
    elif conclusion == "failure":
        result["decision"], result["reason"] = "HOLD", "deterministic_failure_is_authoritative"
    elif conclusion in {None, "skipped"} and age_minutes >= target.max_age_hours * 60:
        result["decision"], result["reason"] = "DISPATCH", "stale_non_success_terminal_state"
    else:
        result["reason"] = "within_window"
    return result

def dispatch_and_verify(repo: str, target: Target, dispatch_epoch: int) -> int:
    _gh(["workflow", "run", target.workflow, "--repo", repo, "--ref", "main"])
    for _ in range(5):
        time.sleep(2)
        runs = list_runs(repo)
        recent = [
            r for r in runs
            if str(r.get("path", "")).lstrip("/") == target.workflow
            and r.get("event") == "workflow_dispatch"
            and parse_time(r["created_at"]).timestamp() >= dispatch_epoch
        ]
        if recent:
            return int(recent[0]["id"])
    raise RuntimeError(f"dispatch accepted but no new run observed: {target.workflow}")

def run(repo: str, output: Path, max_dispatches_per_cycle: int = 2) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    runs = list_runs(repo)
    decisions = [decide(target, runs, now) for target in TARGETS]
    dispatched = 0
    heavy_dispatched = 0
    for item, target in zip(decisions, TARGETS):
        if item["decision"] != "DISPATCH":
            continue
        if dispatched >= max_dispatches_per_cycle:
            item["decision"], item["reason"] = "HOLD", "cycle_dispatch_cap_reached"
            continue
        if target.heavy and heavy_dispatched >= 1:
            item["decision"], item["reason"] = "HOLD", "heavy_dispatch_cap_reached"
            continue
        epoch = int(time.time())
        run_id = dispatch_and_verify(repo, target, epoch)
        item["decision"] = "DISPATCHED"
        item["dispatched_run_id"] = run_id
        dispatched += 1
        if target.heavy:
            heavy_dispatched += 1
    blockers = [
        x for x in decisions
        if x["decision"] == "HOLD" and x["reason"] == "deterministic_failure_is_authoritative"
    ]
    report = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "git_commit": os.environ.get("GITHUB_SHA") or None,
        "repository": repo,
        "control_plane_status": "EXECUTED_WITH_BLOCKERS" if blockers else "EXECUTED",
        "cycle_dispatch_count": dispatched,
        "cycle_heavy_dispatch_count": heavy_dispatched,
        "targets": decisions,
        "deterministic_failure_targets": [x["workflow"] for x in blockers],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--output", default="results/control_plane/autonomous_control_plane.json")
    parser.add_argument("--max-dispatches-per-cycle", type=int, default=2)
    args = parser.parse_args()
    if not args.repo:
        raise SystemExit("GITHUB_REPOSITORY is required")
    report = run(args.repo, ROOT / args.output, args.max_dispatches_per_cycle)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
