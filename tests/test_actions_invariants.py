from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CLOSED_LOOP = ROOT / ".github" / "workflows" / "baseball_closed_loop.yml"
RECOVERY = ROOT / ".github" / "workflows" / "baseball_actions_recovery.yml"
X_RESEARCH = ROOT / ".github" / "workflows" / "baseball_x_research.yml"
AUTOPILOT = ROOT / ".github" / "workflows" / "baseball_9h_autopilot.yml"
SUPERVISOR = ROOT / ".github" / "workflows" / "baseball_24h_supervisor.yml"


def _assert_official_actions_are_immutable(text: str) -> None:
    """Every actions/* reference must be pinned to a full commit SHA."""
    refs = re.findall(r"uses:\s*(actions/[^@\s]+)@([^\s#]+)", text)
    assert refs, "expected at least one official GitHub Action reference"
    for action, ref in refs:
        assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{action} is not pinned to an immutable SHA: {ref}"


def test_closed_loop_keeps_safe_sequential_execution_and_quality_gates():
    text = CLOSED_LOOP.read_text(encoding="utf-8")

    assert "group: baseball-closed-loop" in text
    # An active research run must not be interrupted; GitHub's concurrency
    # group keeps the lifecycle sequential while newer pending runs wait.
    assert "cancel-in-progress: false" in text
    assert "queue: max" not in text
    assert "- cron: '17 0,3,6,9,12,15,18,21 * * *'" in text
    assert "120-minute lifecycle timeout" in text
    _assert_official_actions_are_immutable(text)

    # PIT must be acquired and validated before any OOS research is allowed.
    pit_pos = text.index("- name: Acquire current PIT observations")
    validate_pos = text.index("- name: Validate PIT before research")
    oos_pos = text.index("- name: Run chronological Baseball OOS research")
    assert pit_pos < validate_pos < oos_pos
    assert "if stage.status != 'READY'" in text

    # Research and lifecycle gates must remain fail-closed.
    assert "obj.get('overall_status') != 'SUCCESS'" in text
    assert "obj.get('status') != 'READY'" in text


def test_recovery_is_bounded_and_only_retries_transient_steps():
    text = RECOVERY.read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)

    # The recovery job itself needs a repository context because gh run view
    # resolves the parent run through the current checkout. Keep that setup
    # immutable and credential-free after checkout.
    assert "- name: Checkout recovery repository context" in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert "persist-credentials: false" in text
    assert "fetch-depth: 1" in text

    assert 'if [ "${RUN_ATTEMPT}" -ge 3 ]; then' in text
    assert 'gh run rerun "${RUN_ID}" --failed' in text
    assert "for recovery_attempt in 1 2 3; do" in text
    assert "No retryable transient failure detected; preserving the failure for diagnosis." in text
    assert "Mixed transient and deterministic failures detected; refusing automatic rerun." in text
    assert "non_retryable=$((non_retryable + 1))" in text
    assert 'if [ "${non_retryable}" -ne 0 ]; then' in text

    # Inspection/API failures must not create a second false-red. The parent
    # workflow failure remains authoritative and is preserved for diagnosis.
    assert "failed_steps=''" in text
    assert "for inspect_attempt in 1 2 3; do" in text
    assert "Recovery could not inspect the parent run after 3 attempts; preserving the original failure signal." in text
    assert 'exit 0' in text

    # Keep data/model quality failures out of automatic reruns.
    assert "Install research dependencies" in text
    assert "Acquire current PIT observations" in text
    assert "Run PIT acquisition" in text
    assert "Run chronological Baseball OOS research" not in text
    assert "Enforce lifecycle completion" not in text


def test_x_research_isolated_and_artifact_fail_closed():
    text = X_RESEARCH.read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)

    # X must stay outside the production closed loop and remain optional/research-only.
    assert "permissions:\n  contents: read" in text
    assert "historical_backtest_eligible') is not False" in text
    assert "X source must remain research-only until historical PIT availability is proven" in text

    # A successful collection must prove that its research manifest and PIT snapshot exist.
    assert "test -s data/x/acquisition_runs.jsonl" in text
    assert "test -s data/pit/x_source_snapshots.jsonl" in text
    assert "if-no-files-found: error" in text

    # The immutable action pin must remain in place.
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text


def test_closed_loop_uses_shared_temporal_calibration_contract():
    text = (ROOT / "research" / "closed_loop_execute.py").read_text(encoding="utf-8")
    assert "from evaluation.calibration import fit_temperature" in text
    assert "fit_temperature(p, y).temperature" in text
    assert "fit_temperature(" in text
    assert "atomic_write_json" in text
    assert "source_fingerprints" in text
    assert '"production_approved": False' in text
    assert 'evaluation_only_no_auto_promotion' in text
    assert 'def development_candidate_id(' in text
    assert 'candidate_id=candidate_id' in text


def test_9h_autopilot_hands_off_evidence_to_final_phase_and_hides_no_failures():
    text = AUTOPILOT.read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)

    assert "actions/download-artifact@fa0a91b85d4f404e444e00e005971372dc801d16" in text
    assert "Restore Phase 2 OOS evidence" in text
    assert "Restore Phase 3 candidate evidence" in text
    assert "Verify evidence handoff before final governance" in text
    assert "test -s phase2-evidence/results/checkpoints/npb_walkforward.csv" in text
    assert "test -s phase2-evidence/results/checkpoints/mlb_walkforward.csv" in text
    assert 'test -s phase2-evidence/results/checkpoints/npb_walkforward.version' in text
    assert 'test -s phase2-evidence/results/checkpoints/mlb_walkforward.version' in text
    assert "Phase 2 OOS and Phase 3 candidate evidence handoff verified." in text
    assert "|| true" not in text
    assert "continue-on-error" not in text


def test_npb_production_never_scores_started_games_and_accepts_empty_future_state():
    production = (ROOT / ".github" / "workflows" / "npb-production.yml").read_text(encoding="utf-8")
    source = (ROOT / "production_npb.py").read_text(encoding="utf-8")

    assert "execution_status" in source
    assert "NO_FUTURE_GAMES" in source
    assert 'if r["datetime"] <= now_utc:' in source
    assert 'd["execution_status"] in {"EXECUTED", "BLOCKED_STARTERS", "NO_FUTURE_GAMES"}' in production
    assert 'd["execution_status"] == "NO_FUTURE_GAMES"' in production
    assert 'd["pit_status"] == "PASS"' in production
    assert 'd["starter_gate"] == "PASS"' in production
    _assert_official_actions_are_immutable(production)


def test_24h_supervisor_avoids_deterministic_failure_retry_loop():
    text = SUPERVISOR.read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)

    assert "Deterministic failures must not enter an unbounded retry loop." in text
    assert "cooldown active" in text
    assert "baseball_actions_recovery.yml owns transient failed-job retries" in text
    assert "gh run rerun" not in text
    assert "gh workflow run" in text
    assert "Dispatch verification" in text
    assert "latest_age_minutes" in text
