import pandas as pd

from research.score_low_high_metrics import binary_probability_metrics, low_high_from_scores, score_metrics


def test_score_metrics_and_low_high_labels():
    frame = pd.DataFrame(
        {
            "pred_home_score": [4.0, 2.2, 6.0],
            "pred_away_score": [2.0, 3.1, 1.0],
            "actual_home_score": [4, 2, 7],
            "actual_away_score": [2, 4, 1],
        }
    )
    metrics = score_metrics(frame)
    assert metrics["rows"] == 3.0
    assert metrics["exact_score_rate"] > 0
    labeled = low_high_from_scores(frame, line=7.5)
    assert labeled["realized_low"].tolist() == [1, 1, 0]
    assert labeled["realized_high"].tolist() == [0, 0, 1]


def test_low_high_probability_metrics_are_oos_only():
    frame = pd.DataFrame({"high_prob": [0.8, 0.2, 0.7, 0.4], "actual_high": [1, 0, 0, 1]})
    metrics = binary_probability_metrics(frame, probability="high_prob", actual="actual_high")
    assert metrics["rows"] == 4.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["log_loss"] >= 0.0
