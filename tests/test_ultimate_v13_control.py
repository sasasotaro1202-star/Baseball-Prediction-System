import numpy as np
import pandas as pd

from research.ultimate_v13_control import (
    active_information,
    artifact_gate,
    data_quality,
    feature_reliability,
    meta_label_walk_forward,
    pit_audit,
    regime_transition,
    retrieval,
    robustness_matrix,
    run_control_plane,
    selective_prediction,
    uncertainty_decomposition,
)


def test_pit_fail_closed_and_meta_leakage():
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    df = pd.DataFrame({
        "prediction_time": [base, base + pd.Timedelta(hours=1)],
        "available_at": [base - pd.Timedelta(minutes=1), base + pd.Timedelta(hours=2)],
    })
    out = pit_audit(df)
    assert out["status"] == "FAIL"
    assert any("after_prediction" in x for x in out["blockers"])


def test_retrieval_is_pit_filtered():
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    hist = pd.DataFrame({
        "prediction_time": [base, base + pd.Timedelta(hours=1), base + pd.Timedelta(hours=3)],
        "available_at": [base - pd.Timedelta(minutes=10), base, base + pd.Timedelta(hours=3)],
        "x": [0.0, 1.0, 0.1],
    })
    out = retrieval(hist, {"x": 0.1}, feature_cols=["x"], prediction_time=base + pd.Timedelta(hours=2), k=2)
    assert out["pit_filtered_rows"] == 2
    assert {r["index"] for r in out["neighbors"]}.issubset({"0", "1"})


def test_selective_uncertainty_and_info():
    y = np.array([0, 1, 1, 0])
    p = np.array([[.8,.2],[.4,.6],[.55,.45],[.9,.1]])
    s = selective_prediction(y, p, risk_score=[.1,.9,.2,.1], threshold=.65)
    assert s["coverage"] == .75
    u = uncertainty_decomposition(disagreement=.3, ood=.2, drift=.1, data_quality_score=.9, info_uncertainty=.2)
    assert 0 <= u["total"] <= 1
    info = active_information([{"name":"a","expected_gain":.1,"cost":.02,"failure_risk":.01}])
    assert info["selected"] == "a"


def test_regime_and_reliability():
    tr = regime_transition(["range", "trend", "trend", "high_volatility"], horizon=2)
    assert tr["current"] == "high_volatility"
    df = pd.DataFrame({"a":[1,2,3],"b":[1,np.nan,3]})
    dq = data_quality(df)
    rel = feature_reliability(df,["a","b","missing"])
    assert 0 <= dq["score"] <= 1
    assert rel["missing"] == 0


def test_meta_label_is_walk_forward_only():
    rng = np.random.default_rng(1)
    y = rng.integers(0,2,120)
    p = np.clip(.50 + rng.normal(0,.18,120), .02,.98)
    out = meta_label_walk_forward(y,p,min_train=50)
    assert out["label_is_future_free"] is True
    assert out["coverage"] > 0


def test_full_control_plane_and_artifact_contract():
    report = run_control_plane()
    assert report["promotion_status"] == "HOLD"
    assert report["artifact_gate"]["status"] == "PASS"
    assert report["promotion_gate"]["status"] == "HOLD"
    assert report["promotion_gate"]["auto_promotion"] is False
    assert "FutureFailure" in {x["component"] for x in report["states"]}
    assert "Robustness" in {x["component"] for x in report["states"]}
    gate = artifact_gate(report)
    assert gate["status"] == "PASS"
    rob = robustness_matrix(np.array([0,1,0]), np.array([[.8,.2],[.2,.8],[.7,.3]]))
    assert 0 <= rob["worst_accuracy"] <= 1
