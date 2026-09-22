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
