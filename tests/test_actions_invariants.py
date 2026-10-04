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
    assert "Recovery could not inspect the parent run after 3 attempts; recovery itself is FAILED and the original failure signal remains authoritative." in text
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


def test_npb_production_checks_runtime_gate_before_network_and_dependency_work():
    text = (ROOT / ".github" / "workflows" / "npb-production.yml").read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)

    gate = text.index("- name: Check NPB production runtime gate")
    install = text.index("- name: Install production dependencies")
    pit = text.index("- name: PIT and syntax gate")
    acquire = text.index("- name: Acquire PIT-safe public NPB PBP data")
    run = text.index("- name: Run NPB production prediction")

    assert gate < install < pit < acquire < run
    required_condition = "if: steps.production_gate.outputs.eligible == 'true'"
    assert text.count(required_condition) >= 4
    assert "execution_status" in text
    assert "BLOCKED_PRODUCTION_GATE" in text
    blocked = (ROOT / "scripts" / "write_blocked_npb_output.py").read_text(encoding="utf-8")
    assert '"execution_status": "BLOCKED_PRODUCTION_GATE"' in blocked
    assert "NPB production runtime is not currently eligible" in blocked



def test_chat_async_dispatcher_is_short_lived_and_allowlisted():
    text = (ROOT / ".github" / "workflows" / "baseball_chat_async_dispatch.yml").read_text(encoding="utf-8")

    assert "issue_comment:" in text
    assert "github.event.issue.number == 74" in text
    assert "github.event.comment.user.login == 'sasasotaro1202-star'" in text
    assert "timeout-minutes: 2" in text

    allowed = (
        "/baseball-async closed-loop",
        "/baseball-async candidate-oos",
        "/baseball-async research-24h",
    )
    for command in allowed:
        assert command in text

    assert 'case "${COMMAND}" in' in text
    assert "gh api" in text
    assert "--method POST" in text
    assert "actions/workflows/" in text
    assert "/dispatches\"" in text
    assert "TARGET_WORKFLOW}" in text
    assert '"repos/${GH_REPO}/actions/workflows/${TARGET_WORKFLOW}/dispatches"' in text
    assert "--argjson inputs" in text
    assert 'ref:"main"' in text
    assert 'workflow=${' not in text
    assert 'ref=${' not in text
    assert "No run polling or completion wait is performed." in text
    assert "No matching trusted async command; no workflow was dispatched." in text

    # Arbitrary shell/workflow execution must not be possible through comment text.
    assert 'workflow=${' not in text
    assert "eval " not in text
    assert "bash -c" not in text


def test_candidate_oos_fails_closed_on_incomplete_evidence():
    text = (ROOT / ".github" / "workflows" / "baseball_candidate_oos.yml").read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)

    assert "- name: Verify candidate evidence completeness" in text
    assert 'test -s results/real_data_validation.json' in text
    assert 'test -s results/npb_candidate_development.json' in text
    assert 'test -s results/npb_locked_holdout.json' in text
    assert 'test -s results/mlb_candidate_development.json' in text
    assert 'test -s results/mlb_locked_holdout.json' in text
    assert "if-no-files-found: error" in text
    assert "if-no-files-found: warn" not in text


def test_npb_production_never_scores_started_games_and_accepts_empty_future_state():
    production = (ROOT / ".github" / "workflows" / "npb-production.yml").read_text(encoding="utf-8")
    source = (ROOT / "production_npb.py").read_text(encoding="utf-8")

    assert "execution_status" in source
    assert "NO_FUTURE_GAMES" in source
    assert 'if r["datetime"] <= now_utc:' in source
    assert 'd["execution_status"] in {"EXECUTED", "BLOCKED_PRODUCTION_GATE", "BLOCKED_STARTERS", "NO_FUTURE_GAMES"}' in production
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
    # The dispatch is wrapped by gh_retry for transient GitHub API resilience.
    assert "gh_retry workflow run" in text
    assert "Dispatch verification" in text
    assert "latest_age_minutes" in text
    # The supervisor deliberately avoids a push trigger so PIT-only evidence commits
    # cannot create a queue-flooding control-plane loop. Main-branch drift is checked
    # explicitly by compare/${latest_sha}...${current_main_sha} instead.
    trigger = text.split("permissions:", 1)[0]
    assert "push:" not in trigger
    assert "compare/${latest_sha}...${current_main_sha}" in text
    assert 'startswith("data/pit/")' in text
    assert "main advanced only through PIT evidence commits" in text



def test_24h_keeper_uses_consecutive_failure_streak():
    text = (ROOT / ".github" / "workflows" / "baseball_24h_research_keeper.yml").read_text(encoding="utf-8")
    assert "failure_streak" in text
    assert "consecutive_failure_streak" in text
    assert "sort_by(.createdAt)" in text
    assert "reverse" in text
    assert "reduce .[] as $r" in text
    assert 'elif $r.conclusion == "failure"' in text
    assert ".done = true" in text
    assert "three or more consecutive failed autopilot runs" in text
    assert "recent_failures_24h" not in text


def test_24h_research_autopilot_uses_current_commit_snapshot_and_no_push_trigger():
    text = (ROOT / ".github" / "workflows" / "baseball_24h_research_autopilot.yml").read_text(encoding="utf-8")
    trigger = text.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    assert "schedule:" in trigger
    assert "push:" not in trigger
    assert "research/data-integration-20260928" not in text
    assert "RESEARCH_REF: ${{ github.sha }}" in text
    assert text.count("ref: ${{ github.sha }}") == 5
    assert text.count("research_snapshot_sha=$actual") == 4
    assert text.count("Verify research snapshot") == 4

def test_chat_async_dispatcher_never_polls_workflow_state():
    text = (ROOT / ".github" / "workflows" / "baseball_chat_async_dispatch.yml").read_text(encoding="utf-8")
    dispatch = text[text.index("- name: Dispatch heavy work immediately"):]
    assert "gh run list" not in dispatch
    assert "gh run view" not in dispatch
    assert "sleep " not in dispatch
    assert "workflow run" not in dispatch
    assert "timeout --signal=TERM --kill-after=3s 10s" in dispatch


def test_regression_ci_does_not_cancel_independent_pull_requests():
    text = (ROOT / ".github" / "workflows" / "baseball_regression_tests.yml").read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)

    assert "group: baseball-regression-tests-${{ github.event.pull_request.number || github.ref }}" in text
    assert "  group: baseball-regression-tests\n" not in text
    assert "cancel-in-progress: true" in text


def test_manual_prediction_contract_preserves_runtime_failure_and_json_stdout():
    workflow = (ROOT / ".github" / "workflows" / "baseball_manual_prediction.yml").read_text(encoding="utf-8")
    production = (ROOT / "production_npb.py").read_text(encoding="utf-8")

    assert 'if [ ! -s "${output}" ]; then' in workflow
    assert "preserving the original runtime failure" in workflow
    assert "except json.JSONDecodeError as exc:" in workflow
    assert 'exit "${rc:-1}"' in workflow
    assert 'PRODUCTION_DEGENERACY_DEBUG", json.dumps' in production
    assert ", file=sys.stderr)" in production


def test_pregame_experience_persist_skips_absent_optional_shadow_dir():
    workflow = (ROOT / ".github" / "workflows" / "baseball_60m_pregame_auto.yml").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "pregame_auto.sh").read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(workflow)

    # The optional research_shadow directory must remain absent-safe in the
    # extracted implementation, while the wrapper stays small and immutable.
    assert script.count('for experience_path in data/experience/predictions data/experience/research_shadow; do') == 2
    assert script.count('if [ -d "$experience_path" ]; then') == 2
    assert script.count('git add "$experience_path"') == 2
    assert 'git add data/experience/predictions/ data/experience/research_shadow/' not in script
    assert "bash scripts/pregame_auto.sh" in workflow
    assert len(workflow.splitlines()) <= 90


def test_pregame_zero_job_failure_has_bounded_control_plane_recovery():
    recovery = (ROOT / ".github" / "workflows" / "baseball_actions_recovery.yml").read_text(encoding="utf-8")
    assert "Baseball 60m Pregame Auto Prediction" in recovery
    trigger = recovery.split("permissions:", 1)[0]
    assert "- .github/workflows/baseball_60m_pregame_auto.yml" in trigger
    assert "- Baseball 60m Pregame Auto Prediction" in trigger
    assert "GH_REPO: ${{ github.repository }}" in recovery
    assert "WORKFLOW_NAME: ${{ github.event.workflow_run.name }}" in recovery
    assert 'if [ "${WORKFLOW_NAME}" = "Baseball 60m Pregame Auto Prediction" ] || [ "${WORKFLOW_NAME}" = ".github/workflows/baseball_60m_pregame_auto.yml" ] || [ "${WORKFLOW_PATH}" = ".github/workflows/baseball_60m_pregame_auto.yml" ]; then' in recovery
    assert 'job_count="$(gh_retry run view "${RUN_ID}" --repo "${GH_REPO}" --json jobs --jq \'.jobs | length\')"' in recovery
    assert 'if [ "${job_count}" -eq 0 ]; then' in recovery
    assert "checking 15-minute cooldown" in recovery
    assert 'prior_age_minutes=$(( (now_epoch - prior_created_epoch) / 60 ))' in recovery
    assert 'if [ "${prior_age_minutes}" -lt 15 ]; then' in recovery
    assert "PRE_GAME_ZERO_JOB_COOLDOWN" in recovery
    assert "workflow run baseball_60m_pregame_auto.yml --repo" in recovery
    assert "Pregame recovery verification" in recovery
    assert "PRE_GAME_ZERO_JOB_REDISPATCHED" in recovery
    assert "FAILED_PREGAME_ZERO_JOB_DISPATCH" in recovery


def test_24h_supervisor_recovers_only_latest_zero_job_pregame_failures_with_daily_cap():
    text = (SUPERVISOR).read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)
    assert "PREGAME_WORKFLOW=baseball_60m_pregame_auto.yml" in text
    # The supervisor inspects the complete main-branch run ledger because both
    # scheduled and control-plane startup failures must remain recoverable.
    assert 'actions/runs?branch=main&per_page=100' in text
    assert 'select(.path == (".github/workflows/" + $workflow))' in text
    assert "latest_failure_json" in text
    assert "latest_failure_id" in text
    assert "latest_failure_job_count=-1" in text
    assert "pregame_api" in text
    assert "pregame_dispatch_api" in text
    assert 'actions/runs?event=workflow_dispatch&branch=main&per_page=100' in text
    assert "pregame_recovery_attempts_24h" in text
    assert "pregame_recovery_age_minutes" in text
    assert "latest_recovery_created" in text
    assert "(now - 86400)" in text
    assert "sort_by(.createdAt) | reverse | .[0]" in text
    assert 'if [ "${pregame_recovery_attempts_24h}" -ge 3 ]; then' in text
    assert 'if [ "${pregame_recovery_age_minutes}" -lt 15 ]; then' in text
    assert "dispatch_epoch" in text
    assert "new_pregame_run_id" in text
    assert "--argjson cutoff" in text
    assert "PRE_GAME_ZERO_JOB_COOLDOWN" in text
    assert "PRE_GAME_ZERO_JOB_DAILY_CAP" in text
    assert 'gh_retry workflow run "${PREGAME_WORKFLOW}" --repo "${GH_REPO}" --ref main' in text
    assert "Pregame recovery verification" in text
    assert "PRE_GAME_ZERO_JOB_REDISPATCHED" in text
    assert "FAILED_PREGAME_ZERO_JOB_DISPATCH" in text
    assert "gh run rerun" not in text
    assert "Latest pregame failure is not a zero-job startup failure" in text
    assert "HTTP 4[0-9]{2}" in text
    assert "HTTP (408|429)" in text
    assert "gh deterministic HTTP 4xx; refusing retry." in text
    pregame = (ROOT / ".github" / "workflows" / "baseball_60m_pregame_auto.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in pregame
    assert "recovery_mode:" in pregame
    assert 'default: "manual"' in pregame
    assert "run: bash scripts/pregame_auto.sh" in pregame
    assert "<<'PY'" not in pregame
    assert len(pregame.splitlines()) <= 90
    assert (ROOT / ".github" / "workflows" / "baseball_60m_pregame_auto.yml").is_file()
    assert (ROOT / "scripts" / "pregame_auto.sh").is_file()
    assert "--field recovery_mode=zero_job_startup_recovery" in text




def test_research_lab_caches_are_snapshot_verified_and_file_scoped():
    """Research caches must contain only the verified evidence for their exact SHA."""
    contracts = (
        ("baseball_ultimate_pattern_lab.yml", "ultimate_pattern_lab_${{ matrix.league }}.json", "baseball-ultimate-pattern-v1"),
        ("baseball_extreme_representation_lab.yml", "extreme_representation_lab_${{ matrix.league }}.json", "baseball-extreme-representation-v1"),
        ("baseball_score_distribution_pattern_lab.yml", "score_distribution_pattern_lab_${{ matrix.league }}.json", "baseball-score-distribution-v1"),
    )
    workflow_root = ROOT / ".github" / "workflows"
    for filename, artifact_name, cache_key_prefix in contracts:
        text = (workflow_root / filename).read_text(encoding="utf-8")
        _assert_official_actions_are_immutable(text)
        verify_pos = text.index("- name: Verify main snapshot after evidence generation")
        save_pos = text.index("- name: Save exact")
        assert verify_pos < save_pos, f"{filename} must snapshot-verify before cache save"
        save_block = text[save_pos:text.find("- name: Upload", save_pos)]
        restore_pos = text.index("- name: Restore exact")
        restore_block = text[restore_pos:save_pos]
        assert f"path: results/{artifact_name}" in restore_block
        assert "path: results\n" not in restore_block
        assert f"path: results/{artifact_name}" in save_block
        assert "path: results\n" not in save_block
        expected_key = "key: " + cache_key_prefix + "-${{ matrix.league }}-${{ github.sha }}"
        assert expected_key in restore_block
        assert expected_key in save_block
        assert "cancel-in-progress: false" in text

def test_phase1_candidate_gate_stages_npb_data_before_real_validation():
    text = (ROOT / ".github" / "workflows" / "baseball_phase1_gate.yml").read_text(encoding="utf-8")
    _assert_official_actions_are_immutable(text)

    cache_pos = text.index("- name: Restore NPB historical PBP cache")
    stage_pos = text.index("- name: Stage NPB historical PBP when cache is cold")
    decision_pos = text.index("- name: Validate fail-closed decision contract")

    assert cache_pos < stage_pos < decision_pos
    assert "actions/cache/restore@1bd1e32a3bdc45362d1e726936510720a7c30a57" in text
    assert "gh release download pbp --repo armstjc/Nippon-Baseball-Data-Repository" in text
    assert "if: matrix.league == 'NPB'" in text
    assert 'test "${files}" -ge 7' in text
    assert "timeout-minutes: 20" in text
def test_24h_autopilot_targeted_test_paths_exist():
    """Never let the long-running autopilot reference deleted test modules."""
    workflow = (ROOT / ".github" / "workflows" / "baseball_24h_research_autopilot.yml").read_text(encoding="utf-8")
    paths = sorted(set(re.findall(r"tests/[A-Za-z0-9_.-]+\.py", workflow)))
    assert paths, "expected at least one targeted test path in the 24h autopilot"
    missing = [path for path in paths if not (ROOT / path).is_file()]
    assert not missing, f"24h autopilot references missing test files: {missing}"


def test_gate_workflows_skip_test_only_pushes_to_reduce_duplicate_ci():
    workflow_root = ROOT / ".github" / "workflows"
    for filename in (
        "baseball_research_preflight.yml",
        "baseball_research_readiness.yml",
        "baseball_hardening_fast_gate.yml",
    ):
        text = (workflow_root / filename).read_text(encoding="utf-8")
        trigger = text.split("permissions:", 1)[0]
        assert "paths-ignore:" in trigger
        assert "      - 'tests/**'" in trigger
        assert "push:" in trigger

