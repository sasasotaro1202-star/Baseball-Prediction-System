#!/usr/bin/env python3
"""Monitoring-only health audit for the canonical NPB production workflow.

This module does not select, promote, demote, or modify a prediction model.
It inspects GitHub Actions runtime state and published production artifacts,
and fails closed when the GitHub API cannot be inspected reliably.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_WORKFLOW = "npb-production.yml"
DEFAULT_OUT = Path("results/monitoring/npb_production_runtime_health.json")

# The canonical NPB production job currently has a 90-minute timeout.
# LONG_RUNNING is an early warning; STALE_RISK is aligned with that timeout.
LONG_RUNNING_MINUTES = 60
STALE_RISK_MINUTES = 90
PRODUCTION_ARTIFACT_PREFIX = "npb-production-output-"


def _utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def elapsed_minutes(started_at: Any, now: datetime) -> float | None:
    ts = _utc(started_at)
    if ts is None:
        return None
    return max(0.0, (now.astimezone(timezone.utc) - ts).total_seconds() / 60.0)


def classify_run(status: str, conclusion: str | None, elapsed: float | None) -> str:
    status = str(status or "unknown").strip().lower()
    conclusion = str(conclusion or "").strip().lower()

    if status == "completed":
        if conclusion == "success":
            return "SUCCESS"
        if conclusion == "cancelled":
            return "CANCELLED"
        if conclusion in {"failure", "timed_out", "action_required"}:
            return "FAILURE"
        return "TERMINAL"

    if status in {"queued", "requested", "waiting", "pending"}:
        return "QUEUED"

    if status == "in_progress":
        if elapsed is None:
            return "RUNNING"
        if elapsed >= STALE_RISK_MINUTES:
            return "STALE_RISK"
        if elapsed >= LONG_RUNNING_MINUTES:
            return "LONG_RUNNING"
        return "RUNNING"

    return "UNKNOWN"


def classify_job(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "unknown").strip().lower()
    conclusion = str(job.get("conclusion") or "").strip().lower()

    if status == "completed":
        if conclusion == "success":
            return "SUCCESS"
        if conclusion == "cancelled":
            return "CANCELLED"
        if conclusion in {"failure", "timed_out", "action_required"}:
            return "FAILURE"
        return "TERMINAL"

    if status == "in_progress":
        return "RUNNING"

    if status in {"queued", "requested", "waiting", "pending"}:
        return "QUEUED"

    return "UNKNOWN"


def _api_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Baseball-Prediction-System-runtime-health",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            value = json.load(response)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise RuntimeError(
            f"GITHUB_API_LOOKUP_FAILED:{type(exc).__name__}:{exc}"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError("GITHUB_API_INVALID_OBJECT")
    return value


def _latest_production_run(
    api_base: str,
    repository: str,
    token: str,
    workflow: str,
) -> dict[str, Any]:
    payload = _api_json(
        f"{api_base}/repos/{repository}/actions/workflows/{workflow}/runs"
        "?branch=main&per_page=20",
        token,
    )
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise RuntimeError("GITHUB_API_MISSING_WORKFLOW_RUNS")

    candidates = [
        run for run in runs
        if isinstance(run, dict)
        and str(run.get("head_branch") or "main").strip() == "main"
    ]
    if not candidates:
        raise RuntimeError("PRODUCTION_RUN_NOT_FOUND")

    def sort_key(run: dict[str, Any]) -> tuple[datetime, int]:
        return (
            _utc(run.get("created_at"))
            or datetime.min.replace(tzinfo=timezone.utc),
            int(run.get("id") or 0),
        )

    # Prefer the currently active execution so a queued successor cannot hide
    # an older production run that is still consuming the hot path.
    active = [
        run for run in candidates
        if str(run.get("status") or "").strip().lower() == "in_progress"
    ]
    if active:
        active.sort(key=sort_key, reverse=True)
        return active[0]

    queued = [
        run for run in candidates
        if str(run.get("status") or "").strip().lower()
        in {"queued", "requested", "waiting", "pending"}
    ]
    if queued:
        queued.sort(key=sort_key, reverse=True)
        return queued[0]

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def _current_main_sha(api_base: str, repository: str, token: str) -> str:
    payload = _api_json(
        f"{api_base}/repos/{repository}/git/ref/heads/main",
        token,
    )
    obj = payload.get("object")
    if not isinstance(obj, dict) or not obj.get("sha"):
        raise RuntimeError("GITHUB_API_MAIN_SHA_MISSING")
    return str(obj["sha"])


def _run_jobs(
    api_base: str,
    repository: str,
    token: str,
    run_id: int,
) -> list[dict[str, Any]]:
    payload = _api_json(
        f"{api_base}/repos/{repository}/actions/runs/{run_id}/jobs?per_page=100",
        token,
    )
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("GITHUB_API_MISSING_JOBS")
    return [job for job in jobs if isinstance(job, dict)]


def _run_artifacts(
    api_base: str,
    repository: str,
    token: str,
    run_id: int,
) -> list[dict[str, Any]]:
    payload = _api_json(
        f"{api_base}/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100",
        token,
    )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("GITHUB_API_MISSING_ARTIFACTS")
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def build_health(
    *,
    run: dict[str, Any],
    jobs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    current_main_sha: str,
    now: datetime,
) -> dict[str, Any]:
    run_id = int(run.get("id") or 0)
    if run_id <= 0:
        raise RuntimeError("PRODUCTION_RUN_ID_INVALID")

    run_elapsed = elapsed_minutes(
        run.get("run_started_at") or run.get("created_at"),
        now,
    )
    health_state = classify_run(
        str(run.get("status") or "unknown"),
        str(run.get("conclusion") or ""),
        run_elapsed,
    )

    job_states = {
        str(job.get("name") or "UNKNOWN"): classify_job(job)
        for job in jobs
    }
    active_jobs = sorted(
        name for name, state in job_states.items()
        if state in {"RUNNING", "QUEUED"}
    )
    failed_jobs = sorted(
        name for name, state in job_states.items()
        if state == "FAILURE"
    )

    production_artifacts = [
        artifact
        for artifact in artifacts
        if str(artifact.get("name") or "").startswith(PRODUCTION_ARTIFACT_PREFIX)
        and not bool(artifact.get("expired", False))
    ]

    head_sha = str(run.get("head_sha") or "")
    sha_alignment = bool(
        head_sha and current_main_sha and head_sha == current_main_sha
    )

    return {
        "schema": "baseball-production-runtime-health-v1",
        # PASS means the audit itself completed. It is intentionally distinct
        # from health_state, which is the operational state being observed.
        "status": "PASS",
        "health_state": health_state,
        "monitoring_only": True,
        "promotion_gate": False,
        "generated_at_utc": now.astimezone(timezone.utc).isoformat(),
        "production_run": {
            "run_id": run_id,
            "workflow": str(run.get("name") or DEFAULT_WORKFLOW),
            "status": str(run.get("status") or "unknown"),
            "conclusion": run.get("conclusion"),
            "head_sha": head_sha or None,
            "current_main_sha": current_main_sha or None,
            "head_sha_matches_current_main": sha_alignment,
            "created_at_utc": run.get("created_at"),
            "run_started_at_utc": run.get("run_started_at"),
            "updated_at_utc": run.get("updated_at"),
            "elapsed_minutes": run_elapsed,
            "html_url": run.get("html_url"),
        },
        "jobs": {
            "total": len(jobs),
            "active": active_jobs,
            "failed": failed_jobs,
            "states": dict(sorted(job_states.items())),
        },
        "artifacts": {
            "production_output_present": bool(production_artifacts),
            "production_output_artifact_names": sorted(
                str(item.get("name") or "") for item in production_artifacts
            ),
            "artifact_count": len(artifacts),
        },
        "thresholds": {
            "long_running_minutes": LONG_RUNNING_MINUTES,
            "stale_risk_minutes": STALE_RISK_MINUTES,
        },
        "warnings": [
            warning
            for warning, present in (
                ("main_sha_mismatch", not sha_alignment),
                ("production_output_artifact_missing", not bool(production_artifacts)),
                ("failed_job_present", bool(failed_jobs)),
            )
            if present
        ],
        "interpretation": {
            "audit_status_is_not_model_health": True,
            "health_does_not_promote_or_demote_models": True,
            "stale_risk_is_not_auto_failure": True,
        },
    }


def collect(
    *,
    repository: str,
    token: str,
    api_base: str,
    workflow: str = DEFAULT_WORKFLOW,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    api_base = api_base.rstrip("/")
    current_main_sha = _current_main_sha(api_base, repository, token)
    run = _latest_production_run(api_base, repository, token, workflow)
    run_id = int(run.get("id") or 0)
    jobs = _run_jobs(api_base, repository, token, run_id)
    artifacts = _run_artifacts(api_base, repository, token, run_id)
    return build_health(
        run=run,
        jobs=jobs,
        artifacts=artifacts,
        current_main_sha=current_main_sha,
        now=now,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitoring-only health audit for the canonical NPB production workflow."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--workflow",
        default=os.environ.get("PRODUCTION_WORKFLOW", DEFAULT_WORKFLOW),
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token or not args.repository:
        report = {
            "schema": "baseball-production-runtime-health-v1",
            "status": "BLOCKED_LOOKUP",
            "health_state": "UNKNOWN",
            "monitoring_only": True,
            "promotion_gate": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": "GITHUB_API_CONTEXT_MISSING",
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False))
        return 1

    try:
        report = collect(
            repository=args.repository,
            token=token,
            api_base=args.api_base,
            workflow=args.workflow,
        )
    except Exception as exc:
        report = {
            "schema": "baseball-production-runtime-health-v1",
            "status": "BLOCKED_LOOKUP",
            "health_state": "UNKNOWN",
            "monitoring_only": True,
            "promotion_gate": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False))
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    # Operational warnings remain monitoring-only. Lookup/integrity failures
    # are non-zero so the audit itself cannot silently appear healthy.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
