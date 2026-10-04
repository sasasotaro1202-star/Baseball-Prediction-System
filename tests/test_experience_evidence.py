import pandas as pd

from research.experience_evidence import build_experience_evidence


def test_experience_evidence_is_explicitly_historical_and_not_current_verification(monkeypatch):
    monkeypatch.setenv("BASEBALL_CHECKED_OUT_SHA", "generator-sha-123")
    frame = pd.DataFrame(
        [
            {
                "git_commit": "prediction-sha-a",
                "model": "Production",
                "model_version": "npb-production-v1",
                "feature_version": "features-v2",
                "calibration_version": "cal-v4",
                "target": "NPB",
            },
            {
                "git_commit": "prediction-sha-a",
                "model": "Production",
                "model_version": "npb-production-v1",
                "feature_version": "features-v2",
                "calibration_version": "cal-v4",
                "target": "NPB",
            },
            {
                "git_commit": "prediction-sha-b",
                "model": "Challenger",
                "model_version": "candidate-v9",
                "feature_version": "features-v3",
                "calibration_version": "cal-v5",
                "target": "NPB",
            },
        ]
    )

    evidence = build_experience_evidence(frame)

    assert evidence["scope"] == "HISTORICAL_POSTGAME_EXPERIENCE"
    assert evidence["performance_status"] == (
        "HISTORICAL_ONLY_NOT_CURRENT_MODEL_VERIFICATION"
    )
    assert evidence["generated_from_commit"] == "generator-sha-123"
    assert evidence["prediction_source_commits"] == [
        "prediction-sha-a",
        "prediction-sha-b",
    ]
    assert evidence["prediction_models"] == ["Challenger", "Production"]
    assert evidence["prediction_model_versions"] == [
        "candidate-v9",
        "npb-production-v1",
    ]
    assert evidence["prediction_feature_versions"] == [
        "features-v2",
        "features-v3",
    ]
    assert evidence["prediction_calibration_versions"] == [
        "cal-v4",
        "cal-v5",
    ]
    assert evidence["comparison_to_current_production"]["status"] == "UNVERIFIED"
    assert evidence["comparison_to_current_production"][
        "current_runtime_equivalence"
    ] == "NOT_ESTABLISHED"


def test_experience_evidence_does_not_invent_missing_metadata(monkeypatch):
    monkeypatch.setenv("BASEBALL_CHECKED_OUT_SHA", "generator-sha-456")
    evidence = build_experience_evidence(
        pd.DataFrame([{"model": "Production", "target": "NPB"}])
    )

    assert evidence["generated_from_commit"] == "generator-sha-456"
    assert evidence["prediction_models"] == ["Production"]
    assert evidence["prediction_targets"] == ["NPB"]
    assert evidence["prediction_source_commits"] == []
    assert evidence["prediction_model_versions"] == []
    assert evidence["prediction_feature_versions"] == []
    assert evidence["prediction_calibration_versions"] == []
