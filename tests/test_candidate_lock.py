from dataclasses import asdict

import research.candidates as candidates_module
from research.candidates import CandidateSpec, candidate_fingerprint, lock_candidate


def test_candidate_lock_persists_reproducible_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(candidates_module, "RESULTS", tmp_path / "results")
    spec = CandidateSpec(
        candidate_id="cand-test-001",
        league="NPB",
        objective="win",
        model_version="Logistic",
        feature_version="features-v1",
        development_metrics={"LogLoss": 0.6, "rows": 500},
        selection_reason="development OOS only",
        git_commit="abc123",
        dataset_hash="deadbeef",
    )
    lock = lock_candidate(spec)
    assert lock["holdout_evaluated"] is False
    assert lock["candidate_fingerprint"] == candidate_fingerprint(spec)

    import json
    payload = json.loads((tmp_path / "results" / "npb_candidate_lock.json").read_text())
    assert payload["candidate"] == asdict(spec)
    assert payload["candidate_fingerprint"] == candidate_fingerprint(spec)
    assert payload["holdout_evaluated"] is False


def test_candidate_registry_fails_closed_on_corrupt_history(tmp_path, monkeypatch):
    import research.candidate_registry as registry

    path = tmp_path / "candidate_registry.json"
    path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(registry, "REGISTRY", path)

    try:
        registry._load()
    except RuntimeError as exc:
        assert "candidate registry cannot be read safely" in str(exc)
    else:
        raise AssertionError("corrupt candidate history must not be silently reset")


def test_candidate_registry_rejects_non_list_history(tmp_path, monkeypatch):
    import json
    import research.candidate_registry as registry

    path = tmp_path / "candidate_registry.json"
    path.write_text(json.dumps({"candidate_id": "unexpected"}), encoding="utf-8")
    monkeypatch.setattr(registry, "REGISTRY", path)

    try:
        registry._load()
    except RuntimeError as exc:
        assert "candidate registry must contain a JSON list" in str(exc)
    else:
        raise AssertionError("invalid candidate registry shape must fail closed")
