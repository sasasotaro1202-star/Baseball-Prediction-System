from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_game_script_autoresearch_is_recurring_and_fail_closed():
    text = _read("npb_game_script_autoresearch_canonical.yml")
    assert 'cron: "45 0,6,12,18 * * *"' in text
    assert '  push:' not in text
    assert "cancel-in-progress: false" in text
    assert 'pit_status' in text
    assert 'production_eligible": False' in text
    assert "production_promotion" in text
    assert "HOLD_RESEARCH_ONLY" in text
    assert "source_fingerprint" in text
    assert "code_fingerprint" in text
    assert "prev_code" in text
    assert "game_script_checkpoint.json" in text
    assert "validation_source_fingerprint" in text
    assert "context_source_fingerprint" in text
    assert 'assert obj.get("code_fingerprint")' in text
    assert "SHADOW_REFRESH_ONLY" in text
    assert "--shadow-only" in text


def test_game_script_autoresearch_has_main_snapshot_guard():
    text = _read("npb_game_script_autoresearch_canonical.yml")
    assert 'test "$GITHUB_REF" = "refs/heads/main"' in text
    assert 'test "$current_main" = "$GITHUB_SHA"' in text


def test_game_script_watchdog_is_bounded_and_non_promoting():
    text = _read("npb_game_script_watchdog.yml")
    assert 'cron: "17 */6 * * *"' in text
    assert "actions: write" in text
    assert "165" in text
    assert "90" in text
    assert "1560" in text
    assert 'gh workflow run "$workflow"' in text
    assert 'gh run cancel "$id"' in text


def test_game_script_lab_uses_explicit_token_environment():
    text = (ROOT / ".github" / "workflows" / "baseball_game_script_lab.yml").read_text(encoding="utf-8")
    assert 'GH_TOKEN: ${{ github.token }}' in text
    assert 'export GH_TOKEN="$GITHUB_TOKEN"' not in text


def test_game_script_autoresearch_decision_guard_is_closed_and_well_formed():
    text = _read("npb_game_script_autoresearch_canonical.yml")
    malformed = "if grep -q '^need_research=0'" + "\n"
    assert malformed not in text
    assert "if grep -q '^need_research=0$' results/research_decision.txt; then" in text
