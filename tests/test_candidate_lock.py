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


def test_record_candidate_uses_atomic_history_persistence(tmp_path, monkeypatch):
    import json
    import research.candidate_registry as registry
    from research.validation_pipeline import ValidationRecord

    registry_path = tmp_path / "candidate_registry.json"
    monkeypatch.setattr(registry, "REGISTRY", registry_path)
    monkeypatch.setattr(
        registry,
        "run_validation_pipeline",
        lambda **_: ValidationRecord(
            candidate_id="cand-atomic",
            stage="locked_holdout_evaluated",
            decision="HOLD",
            development={"stage": "candidate_locked", "development_metrics": {"rows": 250}},
            locked_holdout={"stage": "locked_holdout_evaluated", "decision": "HOLD"},
        ),
    )

    record = registry.record_candidate(
        candidate_id="cand-atomic",
        git_commit="abc123",
        feature_version="features-v1",
        model_version="model-v1",
        development_metrics={"rows": 250, "LogLoss": 0.68},
        holdout_baseline={"rows": 250, "LogLoss": 0.70, "Brier": 0.25, "Accuracy": 0.60},
        holdout_candidate={"rows": 250, "LogLoss": 0.69, "Brier": 0.25, "Accuracy": 0.60},
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
        holdout_score_baseline=None,
        holdout_score_candidate=None,
        holdout_hilo_baseline=None,
        holdout_hilo_candidate=None,
        league="NPB",
    )

    assert record.candidate_id == "cand-atomic"
    assert registry_path.exists()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["candidate_id"] == "cand-atomic"
