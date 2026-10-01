import numpy as np
import pandas as pd
import pytest

from research.experience_learning import (
    apply_policy,
    build_policy,
    load_experience,
    replay,
)


def _toy(n: int = 80) -> pd.DataFrame:
    cutoffs = pd.date_range("2026-09-01T00:00:00Z", periods=n, freq="h")
    available = cutoffs + pd.Timedelta(hours=2)
    rows = []
    for i, (cutoff, available_at) in enumerate(zip(cutoffs, available)):
        actual = "HOME_WIN" if i % 4 else "DRAW"
        rows.append(
            {
                "prediction_id": f"p{i}",
                "target": "NPB",
                "prediction_cutoff_utc": cutoff,
                "experience_available_at_utc": available_at,
                "actual_outcome": actual,
                "home_win_pct": 55.0,
                "draw_pct": 20.0,
                "away_win_pct": 25.0,
                "regime": "normal" if i % 2 else "volatile",
                "score_regime": "balanced",
                "model": "production",
                "situation_tags": '["tag_a"]' if i % 3 == 0 else '[]',
            }
        )
    return pd.DataFrame(rows)


def test_load_experience_rejects_impossible_experience_time(tmp_path):
    frame = _toy(10)
    frame.loc[3, "experience_available_at_utc"] = (
        frame.loc[3, "prediction_cutoff_utc"] - pd.Timedelta(minutes=1)
    )
    path = tmp_path / "experience.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="precedes prediction_cutoff_utc"):
        load_experience(path)


def test_load_experience_accepts_unit_probabilities(tmp_path):
    frame = _toy(10)
    path = tmp_path / "experience.csv"
    frame.to_csv(path, index=False)
    loaded = load_experience(path)
    assert len(loaded) == 10


def test_build_policy_excludes_experience_after_cutoff():
    frame = _toy()
    cutoff = pd.Timestamp("2026-09-04T00:00:00Z")
    policy = build_policy(frame, cutoff=cutoff, min_group_rows=5, prior_strength=10)
    expected = int(
        (
            (frame["experience_available_at_utc"] <= cutoff)
            & (frame["prediction_cutoff_utc"] < cutoff)
        ).sum()
    )
    assert policy["matured_rows"] == expected
    assert policy["cutoff_utc"].startswith("2026-09-04T00:00:00")


def test_apply_policy_keeps_probabilities_normalized_and_bounded():
    frame = _toy()
    policy = build_policy(frame, cutoff="2026-09-10T00:00:00Z", min_group_rows=5)
    row = frame.iloc[-1]
    adjusted, source = apply_policy(
        np.array([0.55, 0.20, 0.25]),
        row,
        policy,
    )
    assert source is not None
    assert np.isfinite(adjusted).all()
    assert np.all(adjusted > 0)
    assert np.isclose(adjusted.sum(), 1.0, atol=1e-12)


def test_replay_never_uses_current_row_experience():
    frame = _toy(50)
    summary, ledger = replay(frame, min_group_rows=5)
    assert summary["status"] == "REPLAYED"
    assert len(ledger) == len(frame)
    first = ledger.iloc[0]
    assert int(first["matured_experience_rows"]) == 0

    # The current row's own result is only available two hours after its
    # prediction cutoff, so it must not contribute to the prediction made at
    # that cutoff.
    for _, row in ledger.iterrows():
        pred_time = pd.Timestamp(row["prediction_cutoff_utc"])
        assert int(
            (
                (frame["experience_available_at_utc"] <= pred_time)
                & (frame["prediction_cutoff_utc"] < pred_time)
            ).sum()
        ) == int(row["matured_experience_rows"])


def test_replay_is_deterministic():
    frame = _toy(50)
    a_summary, a_ledger = replay(frame, min_group_rows=5)
    b_summary, b_ledger = replay(frame, min_group_rows=5)
    assert a_summary == b_summary
    pd.testing.assert_frame_equal(a_ledger, b_ledger)
