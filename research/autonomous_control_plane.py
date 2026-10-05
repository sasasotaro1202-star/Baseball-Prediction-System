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
    Target(".github/workflows/baseball-user-prediction-request.yml", 2.0, pending_recover_minutes=30, max_runtime_hours=1.5),
    Target(".github/workflows/baseball_candidate_oos_watchdog.yml", 2.0, pending_recover_minutes=30, max_runtime_hours=0.5),
    Target(".github/workflows/baseball_research_readiness.yml", 6.0, pending_recover_minutes=45, max_runtime_hours=0.75),
    Target(".github/workflows/baseball_research_preflight.yml", 6.0, pending_recover_minutes=45, max_runtime_hours=0.75),
    Target(".github/workflows/baseball_governance_autopilot.yml", 8.0, pending_recover_minutes=30, max_runtime_hours=0.5),
    Target(".github/workflows/project_source_provenance_audit.yml", 30.0, pending_recover_minutes=60, max_runtime_hours=0.5),
    Target(".github/workflows/baseball_closed_loop.yml", 10.0, True, 45, 3.0),
    Target(".github/workflows/baseball_24h_research_autopilot_canonical.yml", 30.0, True, 60, 7.0),
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
    # Daily Game-Script challenger. It remains research-only and is also
    # protected by its own six-hour watchdog; this target gives the project
    # control plane a second bounded recovery path without production writes.
    Target(".github/workflows/npb_game_script_autoresearch_canonical.yml", 30.0, True, 60, 3.0),
    # PR #204 added a separate six-hour Monte Carlo Game-Script Lab. Keep it
    # inside the same autonomous heartbeat so a missed schedule cannot leave a
    # newly-added research lane silently dormant.
    Target(".github/workflows/baseball_game_script_lab.yml", 8.0, True, 60, 5.5),
)

ROOT = Path(__file__).resolve().parents[1]

# GitHub may retain stale/legacy run records after a workflow trigger is changed.
# Such events are never valid scheduler evidence for these canonical autonomous lanes.
SUPPORTED_EVENTS_BY_WORKFLOW = {
    ".github/workflows/baseball_autonomous_control_plane_canonical.yml": frozenset({"schedule", "workflow_dispatch"}),
    ".github/workflows/baseball_24h_research_autopilot_canonical.yml": frozenset({"schedule", "workflow_dispatch"}),
    ".github/workflows/npb_game_script_autoresearch_canonical.yml": frozenset({"schedule", "workflow_dispatch"}),
    ".github/workflows/baseball_24h_research_keeper_canonical.yml": frozenset({"schedule", "workflow_dispatch"}),
    ".github/workflows/baseball_actions_recovery.yml": frozenset({"workflow_run"}),
}


def supported_event(workflow: str, event: str | None) -> bool:
    allowed = SUPPORTED_EVENTS_BY_WORKFLOW.get(workflow)
    if allowed is None or not event:
        return True
    return event in allowed


def workflow_cli_ref(workflow: str) -> str:
    """Return the GitHub CLI workflow identifier for a canonical workflow path."""
    return Path(workflow).name


def _is_transient_gh_failure(message: str) -> bool:
    lowered = message.lower()
    transient_markers = (
        "http 408",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "rate limit",
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "temporary failure",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "eof",
    )
    return any(marker in lowered for marker in transient_markers)


def _gh(args: list[str]) -> str:
    last_error: str | None = None
    for attempt in range(1, 4):
        try:
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
        except subprocess.TimeoutExpired as exc:
            last_error = f"gh timeout after 45s (attempt {attempt}/3): {exc}"
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            last_error = (
                f"gh command failed rc={exc.returncode} "
                f"(attempt {attempt}/3): {detail}"
            )
            if not _is_transient_gh_failure(detail):
                raise RuntimeError(last_error) from exc

        if attempt < 3:
            time.sleep(attempt * 3)

    raise RuntimeError(last_error or "gh command failed without diagnostic output")



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
            workflow_cli_ref(target.workflow),
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

    # A terminal failure with zero jobs is a different failure mode from a
    # real job failure. Keep this explicit so the control plane can recover
    # scheduler/Actions startup failures without hiding deterministic test or
    # research failures. Unknown job counts remain fail-closed.
    latest_by_workflow: dict[str, dict[str, Any]] = {}
    for row in rows:
        workflow = str(row.get("path", ""))
        current = latest_by_workflow.get(workflow)
        if current is None or str(row.get("created_at", "")) > str(current.get("created_at", "")):
            latest_by_workflow[workflow] = row
    for target in TARGETS:
        row = latest_by_workflow.get(target.workflow)
        if not row or row.get("conclusion") != "failure" or not row.get("id"):
            continue
        raw_jobs = _gh([
            "run",
            "view",
            str(row["id"]),
            "--repo",
            repo,
            "--json",
            "jobs",
            "--jq",
            ".jobs | length",
        ]).strip()
        try:
            row["job_count"] = int(raw_jobs)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid job count for failed workflow {target.workflow}: {raw_jobs!r}"
            ) from exc
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
    result = [
        r
        for r in runs
        if str(r.get("path", "")).lstrip("/") == workflow
        and supported_event(workflow, str(r.get("event", "")) or None)
    ]
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
    if conclusion == "skipped" and target.skip_is_healthy:
        result["reason"] = "expected_skipped_state"
    elif conclusion in RECOVERABLE and age_minutes >= 15:
        result["decision"], result["reason"] = "DISPATCH", f"recoverable_terminal_state:{conclusion}"
    elif conclusion == "success" and age_minutes >= target.max_age_hours * 60:
        result["decision"], result["reason"] = "DISPATCH", "stale_success"
    elif conclusion == "failure":
        job_count = latest.get("job_count")
        if job_count == 0 and age_minutes >= 15:
            result["decision"], result["reason"] = "DISPATCH", "startup_failure_no_jobs"
        else:
            result["decision"], result["reason"] = "HOLD", "deterministic_failure_or_unverifiable_startup_state"
            if job_count is not None:
                result["failure_job_count"] = int(job_count)
    elif conclusion == "skipped":
        result["reason"] = "skipped_within_controlled_state"
    elif conclusion is None and age_minutes >= target.max_age_hours * 60:
        result["decision"], result["reason"] = "DISPATCH", "stale_non_success_terminal_state"
    else:
        result["reason"] = "within_window"
    return result

def _active_for(runs: list[dict[str, Any]], workflow: str) -> list[dict[str, Any]]:
    return [r for r in _matching(runs, workflow) if r.get("status") in ACTIVE]

def cancel_run(repo: str, run_id: int) -> None:
    _gh(["run", "cancel", str(run_id), "--repo", repo])
    for _ in range(6):
        time.sleep(2)
        status = _gh([
            "run",
            "view",
            str(run_id),
            "--repo",
            repo,
            "--json",
            "status",
            "--jq",
            ".status",
        ]).strip()
        if status not in ACTIVE:
            return
    raise RuntimeError(f"cancel accepted but run remained active: {run_id}")


def dispatch_and_verify(repo: str, target: Target, dispatch_epoch: int) -> int:
    _gh(["workflow", "run", workflow_cli_ref(target.workflow), "--repo", repo, "--ref", "main"])
    for _ in range(6):
        time.sleep(2)
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
            "10",
            "--json",
            "databaseId,status,conclusion,createdAt,updatedAt,headSha,headBranch,event,path",
        ])
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid dispatch verification JSON for {target.workflow}: {exc}"
            ) from exc
        recent = [
            r
            for r in payload
            if r.get("headBranch") == "main"
            and r.get("event") == "workflow_dispatch"
            and parse_time(r["createdAt"]).timestamp() >= dispatch_epoch
        ]
        if recent:
            return int(recent[0]["databaseId"])
    raise RuntimeError(f"dispatch accepted but no new run observed: {target.workflow}")

def run(repo: str, output: Path, max_dispatches_per_cycle: int = 2) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    runs = list_runs(repo)
    main_sha = current_main_sha(repo)
    decisions = [
        decide(target, runs, now, max_dispatches_per_cycle, main_sha)
        for target in TARGETS
    ]
    dispatched = 0
    heavy_dispatched = 0

    for item, target in zip(decisions, TARGETS):
        if item["decision"] not in {"DISPATCH", "RECOVER"}:
            continue
        if dispatched >= max_dispatches_per_cycle:
            item["decision"], item["reason"] = "HOLD", "cycle_dispatch_cap_reached"
            continue
        if target.heavy and heavy_dispatched >= 1:
            item["decision"], item["reason"] = "HOLD", "heavy_dispatch_cap_reached"
            continue

        if item["decision"] == "RECOVER" and item.get("latest_run_id") is not None:
            cancel_run(repo, int(item["latest_run_id"]))
            item["recovered_run_id"] = int(item["latest_run_id"])

        epoch = int(time.time())
        run_id = dispatch_and_verify(repo, target, epoch)
        item["decision"] = "DISPATCHED"
        item["dispatched_run_id"] = run_id
        dispatched += 1
        if target.heavy:
            heavy_dispatched += 1

    blockers = [
        x
        for x in decisions
        if x["decision"] == "HOLD"
        and x["reason"] == "deterministic_failure_or_unverifiable_startup_state"
    ]
    report = {
        "schema_version": 2,
        "generated_at": now.isoformat(),
        "git_commit": os.environ.get("GITHUB_SHA") or None,
        "current_main_sha": main_sha,
        "repository": repo,
        "control_plane_status": "EXECUTED_WITH_BLOCKERS" if blockers else "EXECUTED",
        "cycle_dispatch_count": dispatched,
        "cycle_heavy_dispatch_count": heavy_dispatched,
        "targets": decisions,
        "deterministic_failure_targets": [x["workflow"] for x in blockers],
        "safety_contract": {
            "production_modified": False,
            "auto_promotion": False,
            "fail_closed": True,
            "max_heavy_dispatches_per_cycle": 1,
        },
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
    if args.max_dispatches_per_cycle < 1:
        raise SystemExit("--max-dispatches-per-cycle must be >= 1")
    report = run(args.repo, ROOT / args.output, args.max_dispatches_per_cycle)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())