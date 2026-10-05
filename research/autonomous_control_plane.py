"""Bounded autonomous control plane for scheduled Baseball GitHub Actions.

The control plane is a scheduler/recovery layer only. It does not promote
models or rewrite production configuration. Expensive research is bounded to
one heavy dispatch per cycle, while scheduler-stuck runs are cancelled and
re-dispatched on the current main snapshot. Workflow history is queried
per-workflow so repository-wide Actions volume cannot hide the latest state.
"""
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
QUEUED = {"queued", "pending", "waiting", "requested"}
RECOVERABLE = {"cancelled", "timed_out", "startup_failure"}

@dataclass(frozen=True)
class Target:
    workflow: str
    max_age_hours: float
    heavy: bool = False
    pending_recover_minutes: int = 45
    max_runtime_hours: float | None = None
    skip_is_healthy: bool = False

TARGETS = (
    Target(".github/workflows/baseball_regression_tests.yml", 12.0, pending_recover_minutes=30, max_runtime_hours=0.75),
    Target(".github/workflows/baseball_v44_compatibility.yml", 12.0, pending_recover_minutes=30, max_runtime_hours=0.75),
    Target(".github/workflows/baseball_24h_supervisor.yml", 1.0, pending_recover_minutes=30, max_runtime_hours=0.25),
    Target(".github/workflows/baseball-production-runtime-health.yml", 2.0, pending_recover_minutes=30, max_runtime_hours=0.5),
    Target(".github/workflows/baseball_candidate_oos_watchdog.yml", 2.0, pending_recover_minutes=30, max_runtime_hours=0.5),
    Target(".github/workflows/baseball_research_readiness.yml", 6.0, pending_recover_minutes=45, max_runtime_hours=0.75),
    Target(".github/workflows/baseball_research_preflight.yml", 6.0, pending_recover_minutes=45, max_runtime_hours=0.75),
    Target(".github/workflows/baseball_governance_autopilot.yml", 8.0, pending_recover_minutes=30, max_runtime_hours=0.5),
    Target(".github/workflows/project_source_provenance_audit.yml", 30.0, pending_recover_minutes=60, max_runtime_hours=0.5),
    Target(".github/workflows/baseball_closed_loop.yml", 10.0, True, 45, 3.0),
    Target(".github/workflows/baseball_24h_research_autopilot.yml", 30.0, True, 60, 7.0),
    Target(".github/workflows/npb-production.yml", 10.0, True, 45, 2.0),
    Target(".github/workflows/npb_prediction_experience_archive.yml", 6.0, pending_recover_minutes=60, max_runtime_hours=0.5, skip_is_healthy=True),
    Target(".github/workflows/npb_experience_reconciliation.yml", 30.0, pending_recover_minutes=60, max_runtime_hours=0.5),
    Target(".github/workflows/npb_experience_learning.yml", 12.0, pending_recover_minutes=60, max_runtime_hours=0.75),
    Target(".github/workflows/baseball_phase1_gate.yml", 30.0, pending_recover_minutes=60, max_runtime_hours=1.0),
    Target(".github/workflows/baseball_universal_readiness.yml", 30.0, pending_recover_minutes=60, max_runtime_hours=0.75),
    Target(".github/workflows/baseball_candidate_oos.yml", 192.0, True, 60, 6.0),
    # Weekly research challenger heartbeat. It is research-only and can never
    # modify the production runtime; the control plane only recovers queued/stale
    # executions and re-dispatches the current main snapshot.
    Target(".github/workflows/npb_game_state_research.yml", 192.0, True, 60, 2.5),
    # Daily Game-Script v4 challenger. It remains research-only and is also
    # protected by its own six-hour watchdog; this target gives the project
    # control plane a second bounded recovery path without production writes.
    Target(".github/workflows/npb_game_script_autoresearch.yml", 30.0, True, 60, 3.0),
    # PR #204 added a separate six-hour Monte Carlo Game-Script Lab. Keep it
    # inside the same autonomous heartbeat so a missed schedule cannot leave a
    # newly-added research lane silently dormant.
    Target(".github/workflows/baseball_game_script_lab.yml", 8.0, True, 60, 5.5),
)

ROOT = Path(__file__).resolve().parents[1]

def _gh(args: list[str]) -> str:
    result = subprocess.run(
        ["gh", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
        env=os.environ.copy(),
    )
    return result.stdout

def list_runs(repo: str) -> list[dict[str, Any]]:
    """Return recent main-branch runs with workflow-scoped history.

    A repository-wide 100-run page is unsafe here because the project has many
    scheduled workflows. High-frequency workflows can evict a low-frequency
    target from that page and cause false no_recent_history decisions.
    Querying each controlled workflow independently makes scheduler state
    deterministic with respect to that target.
    """
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        raw = _gh([
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            target.workflow,
            "--branch",
            "main",
            "--limit",
            "50",
            "--json",
            "databaseId,status,conclusion,createdAt,updatedAt,headSha,headBranch,event,path",
        ])
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid workflow-run JSON for {target.workflow}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise RuntimeError(
                f"unexpected workflow-run payload for {target.workflow}: "
                f"{type(payload).__name__}"
            )
        for item in payload:
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"workflow-run row is not an object: {target.workflow}"
                )
            if item.get("headBranch") != "main":
                continue
            rows.append(
                {
                    "id": item.get("databaseId"),
                    "status": item.get("status"),
                    "conclusion": item.get("conclusion"),
                    "created_at": item.get("createdAt"),
                    "updated_at": item.get("updatedAt"),
                    "head_sha": item.get("headSha"),
                    "head_branch": item.get("headBranch"),
                    "event": item.get("event"),
                    "path": item.get("path") or target.workflow,
                }
            )
    return rows

def current_main_sha(repo: str) -> str:
    value = _gh(["api", f"repos/{repo}/git/ref/heads/main", "--jq", ".object.sha"]).strip()
    if not value:
        raise RuntimeError("main branch SHA could not be resolved")
    return value

def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def dispatch_count(runs: list[dict[str, Any]], workflow: str, now: datetime) -> int:
    cutoff = now - timedelta(hours=24)
    return sum(
        1
        for r in runs
        if str(r.get("path", "")).lstrip("/") == workflow
        and r.get("event") == "workflow_dispatch"
        and parse_time(r["created_at"]) >= cutoff
    )

def _matching(runs: list[dict[str, Any]], workflow: str) -> list[dict[str, Any]]:
    result = [r for r in runs if str(r.get("path", "")).lstrip("/") == workflow]
    result.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return result

def decide(
    target: Target,
    runs: list[dict[str, Any]],
    now: datetime,
    cap: int = 2,
    current_sha: str | None = None,
) -> dict[str, Any]:
    matching = _matching(runs, target.workflow)
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
        latest_active = active[0]
        created = parse_time(latest_active["created_at"])
        age_minutes = max(0.0, (now - created).total_seconds() / 60.0)
        run_sha = str(latest_active.get("head_sha") or "")
        result.update(
            {
                "latest_run_id": latest_active.get("id"),
                "latest_status": latest_active.get("status"),
                "latest_age_minutes": round(age_minutes, 2),
                "active_head_sha": run_sha or None,
                "active_sha_relation": (
                    "CURRENT" if current_sha and run_sha == current_sha else "OLDER_OR_UNKNOWN"
                ),
            }
        )

        if attempts >= cap:
            result["decision"], result["reason"] = "HOLD", "dispatch_cap_reached_while_active"
            return result

        if latest_active.get("status") in QUEUED:
            if current_sha and run_sha and run_sha != current_sha:
                result["decision"], result["reason"] = "RECOVER", "queued_run_on_superseded_sha"
                return result
            if age_minutes >= target.pending_recover_minutes:
                result["decision"], result["reason"] = "RECOVER", "scheduler_stuck_pending"
                return result
            result["reason"] = "active_queued_run"
            return result

        if latest_active.get("status") == "in_progress":
            if (
                target.max_runtime_hours is not None
                and age_minutes >= target.max_runtime_hours * 60.0
            ):
                result["decision"], result["reason"] = "RECOVER", "stale_in_progress_run"
                return result
            result["reason"] = "active_in_progress_run"
            return result

        result["reason"] = "active_run"
        return result