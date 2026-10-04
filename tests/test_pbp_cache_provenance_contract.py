from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSED_LOOP = ROOT / ".github" / "workflows" / "baseball_closed_loop.yml"
MANUAL = ROOT / ".github" / "workflows" / "baseball_manual_prediction.yml"
USER_REQUEST = ROOT / ".github" / "workflows" / "baseball-user-prediction-request.yml"
PHASE1 = ROOT / ".github" / "workflows" / "baseball_phase1_gate.yml"


def _text(path: Path) -> str:
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def test_closed_loop_pbp_cache_is_bound_to_published_release() -> None:
    text = _text(CLOSED_LOOP)
    assert "name: Resolve published NPB PBP release fingerprint" in text
    assert "id: npb-pbp-release" in text
    assert "git/ref/tags/pbp" in text
    assert "steps.npb-pbp-release.outputs.sha" in text
    assert "Never restore across release fingerprints." in text


def test_manual_pbp_cache_is_bound_to_published_release() -> None:
    text = _text(MANUAL)
    assert "git/ref/tags/pbp" in text
    assert "steps.pbp_ref.outputs.sha" in text
    assert "Do not restore a different published PBP release" in text


def test_user_request_pbp_cache_is_bound_to_published_release() -> None:
    text = _text(USER_REQUEST)
    assert "git/ref/tags/pbp" in text
    assert "steps.npb-history-cache-key.outputs.source_tag_sha" in text


def test_phase1_pbp_cache_is_bound_to_published_release() -> None:
    text = _text(PHASE1)
    assert "name: Resolve published NPB PBP release fingerprint" in text
    assert "git/ref/tags/pbp" in text
    assert "steps.npb-pbp-release.outputs.sha" in text
    assert "Raw-data cache identity includes the published release snapshot." in text
