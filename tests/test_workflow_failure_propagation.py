from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_retry_preserves_failed_gh_exit_code():
    text = (ROOT / ".github/workflows/baseball_24h_supervisor.yml").read_text(encoding="utf-8")
    assert 'if output="$(gh "$@" 2>&1)"; then' in text
    assert re.search(r"\\belse\\s*\\n\\s*rc=\\$\\?", text)
    assert 'return "$rc"' in text
    assert 'failed (rc=${rc})' in text


def test_watchdog_retries_all_mutating_or_reading_gh_calls():
    text = (ROOT / ".github/workflows/baseball_closed_loop_watchdog.yml").read_text(encoding="utf-8")
    assert "gh_retry() {" in text
    assert 'gh_retry api ' in text
    assert 'gh_retry run list ' in text
    assert 'gh_retry run cancel ' in text
    assert '\n          current_sha="$(gh api ' not in text
    assert '\n          runs_json="$(gh run list ' not in text
    assert "group: baseball-closed-loop-watchdog" in text
