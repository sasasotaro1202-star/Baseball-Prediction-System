from pathlib import Path

from research.project_governance import (
    source_contract_errors,
    source_provenance_errors,
    runtime_policy_errors,
    workflow_contract_errors,
)


def test_source_contract_accepts_all_85_sections():
    text = "\n".join(f"\n{i}. SECTION {i}" for i in range(1, 86))
    text += "\n" + "\n".join(
        [
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
        ]
    )
    assert source_contract_errors(text) == []


def test_source_contract_rejects_missing_section_and_invariant():
    text = "\n".join(f"{i}. Section {i}" for i in range(1, 85))
    errors = source_contract_errors(text)
    assert any("project_source_sections_invalid" in x for x in errors)
    assert any("project_source_required_text_missing" in x for x in errors)


def test_runtime_policy_is_fail_closed_and_non_promoting():
    payload = {
        "policy": {
            "fail_closed": True,
            "auto_promotion": False,
            "research_fallback": False,
            "runtime_identity_is_recorded": True,
        },
        "runtimes": {},
    }
    assert runtime_policy_errors(payload) == []


def test_runtime_policy_rejects_auto_promotion():
    payload = {
        "policy": {
            "fail_closed": True,
            "auto_promotion": True,
            "research_fallback": False,
            "runtime_identity_is_recorded": True,
        },
        "runtimes": {},
    }
    assert "production_runtime_auto_promotion_must_be_false" in runtime_policy_errors(payload)


def test_workflow_contract_rejects_failure_masking_and_unpinned_actions(tmp_path: Path):
    bad = """
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - run: false || true
"""
    errors = workflow_contract_errors(bad, tmp_path / "bad.yml")
    assert any("workflow_failure_masking" in x for x in errors)
    assert any("workflow_failure_masking_or_true" in x for x in errors)
    assert any("workflow_unpinned_action" in x for x in errors)


def test_source_file_is_present():
    source = Path("PROJECT_SOURCE.md")
    assert source.is_file()
    assert source.stat().st_size > 0


def test_source_section_parser_ignores_numbered_lists():
    from research.project_governance import SECTION_RE
    text = "71. COST FIREWALL\n\n1. verified free\n2. free quota\n3. OSS/local\n\n⸻\n\n72. SECURITY / DATA GOVERNANCE\n"
    assert [int(m[0]) for m in SECTION_RE.findall(text)] == [71, 72]

def test_source_provenance_accepts_matching_hash(tmp_path: Path):
    import hashlib
    import json
    source = tmp_path / "PROJECT_SOURCE.md"
    provenance = tmp_path / "PROJECT_SOURCE_PROVENANCE.json"
    payload = "canonical source\n"
    source.write_text(payload, encoding="utf-8")
    provenance.write_text(
        json.dumps({
            "schema_version": 1,
            "source_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "source_bytes": len(payload.encode("utf-8")),
        }),
        encoding="utf-8",
    )
    assert source_provenance_errors(tmp_path) == []


def test_source_provenance_rejects_drift(tmp_path: Path):
    import hashlib
    import json
    source = tmp_path / "PROJECT_SOURCE.md"
    provenance = tmp_path / "PROJECT_SOURCE_PROVENANCE.json"
    source.write_text("changed\n", encoding="utf-8")
    provenance.write_text(
        json.dumps({
            "schema_version": 1,
            "source_sha256": hashlib.sha256(b"canonical\n").hexdigest(),
            "source_bytes": len(b"canonical\n"),
        }),
        encoding="utf-8",
    )
    errors = source_provenance_errors(tmp_path)
    assert errors and errors[0].startswith("project_source_sha256_mismatch:")


def test_critical_automation_files_are_in_governance_contracts():
    from research.project_governance import WORKFLOW_CONTRACTS
    for path in (
        ".github/workflows/baseball_governance_autopilot.yml",
        ".github/workflows/npb_prediction_experience_archive.yml",
        ".github/workflows/npb_experience_reconciliation.yml",
        ".github/workflows/npb_experience_learning.yml",
    ):
        assert path in WORKFLOW_CONTRACTS


def test_governance_workflow_expressions_are_not_backslash_escaped():
    workflow = Path(".github/workflows/baseball_governance_autopilot.yml").read_text(encoding="utf-8")
    expression = "$" + "{{"
    assert "\\" + expression not in workflow
    assert expression + " github.token }}" in workflow
    assert expression + " github.repository }}" in workflow
    assert expression + " github.sha }}" in workflow


def test_pending_actions_status_is_treated_as_active(monkeypatch):
    from research import project_governance as governance

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "current")
    class Fake:
        pass

    def fake_gh_json(args):
        return {
            "workflow_runs": [{
                "path": ".github/workflows/baseball_governance_autopilot.yml",
                "status": "pending",
                "conclusion": None,
                "created_at": "2026-10-04T09:00:00Z",
                "updated_at": "2026-10-04T09:00:00Z",
                "head_sha": "current",
                "id": 1,
                "run_number": 1,
            }]
        }

    monkeypatch.setattr(governance, "_gh_json", fake_gh_json)
    from datetime import datetime, timezone
    report = governance.action_health("owner/repo", datetime(2026, 10, 4, 9, 30, tzinfo=timezone.utc))
    assert report["workflows"][".github/workflows/baseball_governance_autopilot.yml"]["state"] == "HEALTHY"


def test_phase1_and_universal_readiness_are_monitored():
    from research.project_governance import WORKFLOW_CONTRACTS
    assert ".github/workflows/baseball_phase1_gate.yml" in WORKFLOW_CONTRACTS
    assert ".github/workflows/baseball_universal_readiness.yml" in WORKFLOW_CONTRACTS


def test_autonomous_control_plane_is_governed():
    from research.project_governance import WORKFLOW_CONTRACTS
    assert ".github/workflows/baseball_autonomous_control_plane.yml" in WORKFLOW_CONTRACTS


def test_no_run_is_deferred_when_autonomous_control_plane_is_active(monkeypatch):
    from datetime import datetime, timezone
    from research import project_governance as governance

    monkeypatch.setenv("GITHUB_SHA", "current")

    def fake_gh_json(args):
        return {
            "workflow_runs": [
                {
                    "path": ".github/workflows/baseball_autonomous_control_plane.yml",
                    "status": "in_progress",
                    "conclusion": None,
                    "created_at": "2026-10-04T09:50:00Z",
                    "updated_at": "2026-10-04T09:55:00Z",
                    "head_sha": "current",
                    "id": 100,
                    "run_number": 10,
                }
            ]
        }

    monkeypatch.setattr(governance, "_gh_json", fake_gh_json)
    report = governance.action_health("owner/repo", datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc))
    entry = report["workflows"][".github/workflows/baseball_closed_loop.yml"]
    assert entry["state"] == "DEFERRED"
    assert "actions_no_recent_run:.github/workflows/baseball_closed_loop.yml" in report["deferred"]
    assert not any("baseball_closed_loop.yml" in x for x in report["blockers"])


def test_expected_skipped_archive_is_not_failed(monkeypatch):
    from datetime import datetime, timezone
    from research import project_governance as governance

    monkeypatch.setenv("GITHUB_SHA", "current")

    def fake_gh_json(args):
        return {
            "workflow_runs": [
                {
                    "path": ".github/workflows/baseball_autonomous_control_plane.yml",
                    "status": "in_progress",
                    "conclusion": None,
                    "created_at": "2026-10-04T09:50:00Z",
                    "updated_at": "2026-10-04T09:55:00Z",
                    "head_sha": "current",
                    "id": 100,
                    "run_number": 10,
                },
                {
                    "path": ".github/workflows/npb_prediction_experience_archive.yml",
                    "status": "completed",
                    "conclusion": "skipped",
                    "created_at": "2026-10-04T09:00:00Z",
                    "updated_at": "2026-10-04T09:00:00Z",
                    "head_sha": "current",
                    "id": 101,
                    "run_number": 101,
                },
            ]
        }

    monkeypatch.setattr(governance, "_gh_json", fake_gh_json)
    report = governance.action_health("owner/repo", datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc))
    entry = report["workflows"][".github/workflows/npb_prediction_experience_archive.yml"]
    assert entry["state"] == "HEALTHY"
    assert not any("npb_prediction_experience_archive.yml" in x for x in report["blockers"])


def test_autonomous_control_plane_is_single_heartbeat_not_workflow_run_driven():
    workflow = Path(".github/workflows/baseball_autonomous_control_plane.yml").read_text(encoding="utf-8")
    assert "cron: '*/15 * * * *'" in workflow
    assert "workflow_run:" not in workflow
    assert "research.autonomous_control_plane" in workflow


def test_actions_recovery_is_singleton():
    from research.project_governance import WORKFLOW_CONTRACTS

    path = ".github/workflows/baseball_actions_recovery.yml"
    contract = WORKFLOW_CONTRACTS[path]
    assert contract["monitor"] is False
    assert "group: baseball-actions-recovery" in contract["required"]
    assert "cancel-in-progress: true" in contract["required"]


def test_actions_recovery_covers_zero_job_research_workflows():
    workflow = Path(".github/workflows/baseball_actions_recovery.yml").read_text(encoding="utf-8")
    assert "NPB Game-Script Auto Research" in workflow
    assert ".github/workflows/npb_game_script_autoresearch.yml" in workflow
    assert "Baseball Game Script Lab" in workflow
    assert ".github/workflows/baseball_game_script_lab.yml" in workflow
    assert ".github/workflows/baseball_24h_research_autopilot.yml" in workflow
    assert 'case "${WORKFLOW_PATH}" in' in workflow
    assert 'case "\\${WORKFLOW_PATH}" in' not in workflow
    assert "ZERO_JOB_COOLDOWN" in workflow
    assert "ZERO_JOB_REDISPATCHED" in workflow
