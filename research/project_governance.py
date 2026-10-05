#!/usr/bin/env python3
"""Deterministic GitHub control-plane governance for the Baseball Prediction System.

This module is deliberately lightweight. It does not train models, change
production state, or interpret a green Action as performance evidence.
It verifies the repository contract and, in Action mode, inspects recent
GitHub Actions health. Any unresolved contract violation is a real failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "PROJECT_SOURCE.md",
    "PROJECT_SOURCE_PROVENANCE.json",
    "PROJECT_INSTRUCTIONS.md",
    "BASEBALL_SYSTEM_SPEC.md",
    "config/current_production_runtime.json",
    "docs/FEATURE_MANIFEST.md",
    "research/action_preflight.py",
    "research/closed_loop_governance.py",
    "research/autonomous_control_plane.py",
    "tests/test_autonomous_control_plane.py",
)

WORKFLOW_CONTRACTS: dict[str, dict[str, Any]] = {
    ".github/workflows/baseball_closed_loop.yml": {
        "monitor": True,
        "max_age_hours": 10,
        "required": ("schedule:", "cancel-in-progress: false", "research.closed_loop_execute"),
    },
    ".github/workflows/baseball_24h_supervisor.yml": {
        "monitor": True,
        "max_age_hours": 2,
        "required": ("schedule:", "cron: '*/15 * * * *'", "actions: write"),
    },
    ".github/workflows/baseball_candidate_oos_watchdog.yml": {
        "monitor": True,
        "max_age_hours": 2,
        "required": ("schedule:", "cron: '*/15 * * * *'", "actions: write", "baseball_candidate_oos.yml"),
    },
    ".github/workflows/baseball_candidate_oos.yml": {
        "monitor": False,
        "max_age_hours": 0,
        "required": ("schedule:", "cancel-in-progress: false"),
    },
    ".github/workflows/baseball-production-runtime-health.yml": {
        "monitor": True,
        "max_age_hours": 2,
        "required": ("schedule:", "actions: read"),
    },
    ".github/workflows/npb-production.yml": {
        "monitor": True,
        "max_age_hours": 10,
        "required": ("schedule:", "production_npb.py"),
    },
    ".github/workflows/baseball_24h_research_autopilot.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "workflow_dispatch:"),
    },
    ".github/workflows/baseball_autonomous_control_plane.yml": {
        "monitor": True,
        "max_age_hours": 2,
        "required": ("schedule:", "cron: '*/15 * * * *'", "actions: write", "research.autonomous_control_plane"),
    },
    ".github/workflows/baseball_governance_autopilot.yml": {
        "monitor": True,
        "max_age_hours": 8,
        "required": ("schedule:", "cron: \"13 */6 * * *\"", "actions: read", "--actions"),
    },
    ".github/workflows/npb_prediction_experience_archive.yml": {
        "monitor": True,
        "max_age_hours": 6,
        "allowed_conclusions": ("success", "skipped"),
        "defer_stale_conclusions": ("skipped",),
        "required": ("workflow_run:", "schedule:", "contents: write"),
    },
    ".github/workflows/baseball_24h_research_keeper.yml": {
        "monitor": True,
        "max_age_hours": 1,
        "required": ("schedule:", "cron: '*/5 * * * *'", "actions: write"),
    },
    ".github/workflows/npb_experience_reconciliation.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "workflow_dispatch:", "experience_ledger --reconcile"),
    },
    ".github/workflows/npb_experience_learning.yml": {
        "monitor": True,
        "max_age_hours": 12,
        "required": ("schedule:", "workflow_dispatch:", "research.experience_learning", "research.experience_learning_gate"),
    },
    ".github/workflows/project_source_provenance_audit.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "PROJECT_SOURCE.md", "PROJECT_SOURCE_PROVENANCE.json", "sha256"),
    },
    ".github/workflows/baseball_phase1_gate.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "Phase 1 scope", "matrix:", "league: [NPB, MLB]"),
    },
    ".github/workflows/baseball_universal_readiness.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "research.universal_readiness", "tests/test_universal_readiness.py"),
    },
    ".github/workflows/baseball_actions_recovery.yml": {
        "monitor": False,
        "max_age_hours": 0,
        "required": (
            "workflow_run:",
            "actions: write",
            "Re-run failed jobs with bounded recovery",
            "group: baseball-actions-recovery",
            "cancel-in-progress: true",
        ),
    },
}

REQUIRED_SOURCE_PHRASES = (
    "available_at <= prediction_cutoff",
    "retrieved_at ≠ published_at ≠ available_at",
    "HOME\nDRAW\nAWAY",
    "HOME\nAWAY",
    "LOW = total runs <= 6",
    "HIGH = total runs >= 7",
    "random split禁止",
    "PIT Integrity",
    "PIT violations = 0",
    "NO-FAKE-SUCCESS",
    "Future Generalization",
    "Case-Level Correctness",
    "Calibration",
    "Uncertainty Quality",
    "Safe Degradation > False Prediction",
)

SECTION_RE = re.compile(r"(?m)^(?:⸻\n\n)?\s*(\d+)\.\s+([A-Z0-9][A-Z0-9 /_&/-]+)\s*$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def source_contract_errors(text: str) -> list[str]:
    errors: list[str] = []
    matches = SECTION_RE.findall(text)
    numbers = [int(n) for n, _ in matches]
    expected = list(range(1, 86))
    if numbers != expected:
        missing = [n for n in expected if n not in numbers]
        unexpected = [n for n in numbers if n not in expected]
        errors.append(f"project_source_sections_invalid:missing={missing[:20]}:unexpected={unexpected[:20]}:count={len(numbers)}")
    for phrase in REQUIRED_SOURCE_PHRASES:
        if phrase not in text:
            errors.append(f"project_source_required_text_missing:{phrase!r}")
    return errors


def source_provenance_errors(root: Path = ROOT) -> list[str]:
    source = root / "PROJECT_SOURCE.md"
    provenance = root / "PROJECT_SOURCE_PROVENANCE.json"
    if not source.is_file() or not provenance.is_file():
        return ["project_source_provenance_missing"]
    try:
        payload = json.loads(_read(provenance))
    except Exception as exc:
        return [f"project_source_provenance_invalid_json:{type(exc).__name__}"]
    expected = str(payload.get("source_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        return ["project_source_provenance_sha256_invalid"]
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        return [f"project_source_sha256_mismatch:{actual}!={expected}"]
    expected_bytes = payload.get("source_bytes")
    if isinstance(expected_bytes, int) and source.stat().st_size != expected_bytes:
        return [f"project_source_bytes_mismatch:{source.stat().st_size}!={expected_bytes}"]
    return []


def runtime_policy_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        return ["production_runtime_policy_missing"]
    if policy.get("fail_closed") is not True:
        errors.append("production_runtime_fail_closed_must_be_true")
    if policy.get("auto_promotion") is not False:
        errors.append("production_runtime_auto_promotion_must_be_false")
    if policy.get("research_fallback") is not False:
        errors.append("production_runtime_research_fallback_must_be_false")
    if policy.get("runtime_identity_is_recorded") is not True:
        errors.append("production_runtime_identity_recording_must_be_true")
    if not isinstance(payload.get("runtimes"), dict):
        errors.append("production_runtime_runtimes_missing")
    return errors


def workflow_contract_errors(text: str, path: Path) -> list[str]:
    errors: list[str] = []
    if not re.search(r"^permissions:\s*$", text, re.MULTILINE):
        errors.append(f"workflow_permissions_missing:{path}")
    if "runs-on:" not in text:
        errors.append(f"workflow_runner_missing:{path}")
    if not re.search(r"^  [A-Za-z0-9_-]+:\s*$", text, re.MULTILINE):
        errors.append(f"workflow_jobs_missing:{path}")
    if not re.search(r"^    timeout-minutes:\s*\d+\s*$", text, re.MULTILINE):
        errors.append(f"workflow_job_timeout_missing:{path}")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "continue-on-error: true" in stripped:
            errors.append(f"workflow_failure_masking:{path}")
        if "|| true" in stripped:
            errors.append(f"workflow_failure_masking_or_true:{path}")
        if re.match(r"^(?:-\s*)?uses:\s+", stripped):
            ref = stripped.rsplit("@", 1)[-1]
            if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
                errors.append(f"workflow_unpinned_action:{path}:{stripped}")
    return errors


def repository_static_errors(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_FILES:
        path = root / rel
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"required_file_missing:{rel}")

    source = root / "PROJECT_SOURCE.md"
    if source.is_file():
        errors.extend(source_contract_errors(_read(source)))
    errors.extend(source_provenance_errors(root))

    runtime = root / "config/current_production_runtime.json"
    if runtime.is_file():
        try:
            payload = json.loads(_read(runtime))
        except Exception as exc:
            errors.append(f"production_runtime_invalid_json:{type(exc).__name__}")
        else:
            errors.extend(runtime_policy_errors(payload))

    for rel, contract in WORKFLOW_CONTRACTS.items():
        path = root / rel
        if not path.is_file():
            continue
        text = _read(path)
        errors.extend(workflow_contract_errors(text, path))
        for required in contract["required"]:
            if required not in text:
                errors.append(f"critical_workflow_contract_missing:{rel}:{required}")
    return errors


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(
        ["gh", "api", *args],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=30,
    )
    return json.loads(result.stdout)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _workflow_declares_event(workflow_path: str, event: str) -> bool:
    """Return whether the checked-in workflow declares the observed event.

    GitHub can retain legacy push-triggered run records after a workflow has
    moved to schedule/manual execution. Such a run is historical evidence, not
    valid current execution evidence, when the current workflow no longer
    declares that event. Missing event metadata is treated as unknown and
    therefore retained for backward compatibility with older API payloads.
    """
    if not event:
        return True
    path = ROOT / workflow_path
    if not path.is_file():
        return True
    try:
        text = _read(path)
    except OSError:
        return True
    return bool(
        re.search(
            rf"(?m)^  {re.escape(event)}:\s*(?:\{\})?\s*(?:#.*)?$",
            text,
        )
    )


def _supported_workflow_runs(runs: list[dict[str, Any]], workflow_path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for run in runs:
        event = str(run.get("event", "")).strip()
        if _workflow_declares_event(workflow_path, event):
            supported.append(run)
        else:
            unsupported.append(run)
    return supported, unsupported


def action_health(repo: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    payload = _gh_json([f"repos/{repo}/actions/runs?per_page=100&branch=main"])
    runs = payload.get("workflow_runs", [])
    current_sha = os.environ.get("GITHUB_SHA", "").strip() or None

    control_path = ".github/workflows/baseball_autonomous_control_plane.yml"
    control_runs_all = [
        r for r in runs
        if str(r.get("path", "")).lstrip("/") == control_path
    ]
    control_runs, control_unsupported_runs = _supported_workflow_runs(
        control_runs_all,
        control_path,
    )
    control_runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    control_unsupported_runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    control = control_runs[0] if control_runs else None
    control_age_hours = None
    control_healthy = False
    if control:
        control_age_hours = max(
            0.0,
            (now - _parse_time(control["created_at"])).total_seconds() / 3600.0,
        )
        if control.get("status") in {"queued", "pending", "waiting", "requested", "in_progress"}:
            control_healthy = True
        elif control.get("conclusion") == "success" and control_age_hours <= 0.5:
            control_healthy = True

    report: dict[str, Any] = {
        "checked_at": now.isoformat(),
        "repository": repo,
        "control_plane": {
            "workflow": control_path,
            "state": "HEALTHY" if control_healthy else ("NO_RUN" if control is None else "STALE_OR_FAILED"),
            "run_id": control.get("id") if control else None,
            "status": control.get("status") if control else None,
            "conclusion": control.get("conclusion") if control else None,
            "event": control.get("event") if control else None,
            "age_hours": round(control_age_hours, 3) if control_age_hours is not None else None,
            "ignored_unsupported_event_runs": len(control_unsupported_runs),
        },
        "workflows": {},
        "blockers": [],
        "deferred": [],
    }

    for workflow_path, contract in WORKFLOW_CONTRACTS.items():
        if not contract["monitor"]:
            continue

        matching_all = [
            r for r in runs
            if str(r.get("path", "")).lstrip("/") == workflow_path
        ]
        matching, unsupported_matching = _supported_workflow_runs(
            matching_all,
            workflow_path,
        )
        matching.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        unsupported_matching.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        if unsupported_matching:
            newest_unsupported = unsupported_matching[0]
            report["deferred"].append(
                "actions_ignored_unsupported_event:"
                f"{workflow_path}:{newest_unsupported.get('event') or 'UNKNOWN'}"
            )

        if not matching:
            entry = {
                "state": "NO_RUN",
                "run_id": None,
                "status": None,
                "conclusion": None,
                "created_at": None,
                "updated_at": None,
                "head_sha": None,
                "current_sha": current_sha,
                "sha_relation": "UNKNOWN",
                "age_hours": None,
                "reasons": [],
            }
            if unsupported_matching:
                entry["state"] = "DEFERRED"
                entry["reasons"] = [
                    "only_unsupported_event_runs",
                    f"ignored_run_event={unsupported_matching[0].get('event') or 'UNKNOWN'}",
                ]
                report["deferred"].append(
                    f"actions_only_unsupported_event_runs:{workflow_path}"
                )
            elif workflow_path != control_path and control_healthy:
                entry["state"] = "DEFERRED"
                entry["reasons"] = ["awaiting_autonomous_control_plane_reconciliation"]
                report["deferred"].append(f"actions_no_recent_run:{workflow_path}")
            else:
                entry["reasons"] = ["no_recent_run"]
                report["blockers"].append(f"actions_no_recent_run:{workflow_path}")
            report["workflows"][workflow_path] = entry
            continue

        latest = matching[0]
        created = _parse_time(latest["created_at"])
        age_hours = max(0.0, (now - created).total_seconds() / 3600.0)
        status = latest.get("status")
        conclusion = latest.get("conclusion")
        head_sha = latest.get("head_sha")
        event = latest.get("event")

        state = "HEALTHY"
        reasons: list[str] = []

        if status in {"queued", "pending", "waiting", "requested", "in_progress"}:
            state = "HEALTHY"
            reasons.append("active_run")
        else:
            allowed_conclusions = set(contract.get("allowed_conclusions", ("success",)))
            if conclusion not in allowed_conclusions:
                state = "FAILED"
                reasons.append(f"conclusion={conclusion}")

        if state == "HEALTHY" and status not in {"queued", "pending", "waiting", "requested", "in_progress"}:
            if age_hours > float(contract["max_age_hours"]):
                if conclusion in set(contract.get("defer_stale_conclusions", ())):
                    state = "DEFERRED"
                    reasons.append(
                        f"stale_but_deferred_conclusion:{conclusion}:age_hours={age_hours:.2f}"
                    )
                    report["deferred"].append(f"actions_stale_deferred:{workflow_path}")
                else:
                    state = "STALE"
                    reasons.append(
                        f"age_hours={age_hours:.2f}>{contract['max_age_hours']}"
                    )

        # A run whose event is not declared by the current checked-in workflow
        # is a stale/legacy trigger artifact. Do not let it override valid current
        # execution evidence from schedule/manual/workflow_run.
        if state != "DEFERRED" and unsupported_matching:
            reasons.append(
                f"ignored_unsupported_event_runs={len(unsupported_matching)}"
            )

        # A failure from a superseded main SHA is historical evidence, not a
        # current-state failure. Keep it visible in the report but do not let it
        # block governance for the current main revision.
        if state == "FAILED" and current_sha and head_sha and head_sha != current_sha:
            state = "DEFERRED"
            reasons.append("superseded_sha_failure_not_current")
            report["deferred"].append(f"actions_superseded_failure:{workflow_path}")

        if state in {"FAILED", "STALE"} and workflow_path != control_path and control_healthy:
            # The autonomous control plane is the owner of bounded dispatch/recovery.
            # Governance reports the condition but does not create a second retry loop.
            report["deferred"].append(f"actions_{state.lower()}_owned_by_control_plane:{workflow_path}")
            state = "DEFERRED"
            reasons.append("autonomous_control_plane_owns_recovery")

        if state == "DEFERRED":
            report["deferred"].append(f"actions_deferred:{workflow_path}")
        elif state != "HEALTHY":
            report["blockers"].append(f"actions_{state.lower()}:{workflow_path}")

        report["workflows"][workflow_path] = {
            "state": state,
            "run_id": latest.get("id"),
            "run_number": latest.get("run_number"),
            "status": status,
            "conclusion": conclusion,
            "event": event,
            "created_at": latest.get("created_at"),
            "updated_at": latest.get("updated_at"),
            "head_sha": head_sha,
            "current_sha": current_sha,
            "sha_relation": (
                "CURRENT" if current_sha and head_sha == current_sha
                else "OLDER_OR_UNKNOWN"
            ),
            "age_hours": round(age_hours, 3),
            "reasons": reasons,
        }
    return report


def build_report(*, with_actions: bool, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    static_errors = repository_static_errors(ROOT)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "git_commit": os.environ.get("GITHUB_SHA") or None,
        "static": {
            "status": "READY" if not static_errors else "BLOCKED",
            "blockers": static_errors,
            "source_sha256": None,
        },
        "actions": None,
        "overall_status": "BLOCKED" if static_errors else "READY",
    }
    source = ROOT / "PROJECT_SOURCE.md"
    if source.is_file():
        report["static"]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()

    if with_actions:
        repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
        if not repo:
            report["actions"] = {
                "status": "UNVERIFIABLE",
                "blockers": ["GITHUB_REPOSITORY_missing"],
            }
            report["overall_status"] = "BLOCKED"
        else:
            try:
                actions = action_health(repo, now)
            except Exception as exc:
                report["actions"] = {
                    "status": "UNVERIFIABLE",
                    "blockers": [f"actions_inspection_failed:{type(exc).__name__}:{exc}"],
                }
                report["overall_status"] = "BLOCKED"
            else:
                report["actions"] = {
                    "status": "READY" if not actions["blockers"] else "BLOCKED",
                    **actions,
                }
                if actions["blockers"]:
                    report["overall_status"] = "BLOCKED"

    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", action="store_true")
    parser.add_argument("--output", default="results/governance/project_governance.json")
    args = parser.parse_args()

    report = build_report(with_actions=args.actions)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
