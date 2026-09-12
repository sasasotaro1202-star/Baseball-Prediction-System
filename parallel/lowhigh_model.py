"""Low/High total-score probability (baseball). Low<=6, High>=7 by default."""
from parallel.score_distribution import _poisson_pmf
from parallel.config import LOWHIGH_THRESHOLD


def low_high_probabilities(home_lambda, away_lambda, threshold=LOWHIGH_THRESHOLD, max_total=30):
    total_probs = [0.0] * (max_total + 1)
    for total in range(max_total + 1):
        s = 0.0
        for h in range(0, total + 1):
            a = total - h
            if a > max_total:
                continue
            s += _poisson_pmf(h, home_lambda) * _poisson_pmf(a, away_lambda)
        total_probs[total] = s
    low_prob = sum(total_probs[: threshold + 1])
    high_prob = sum(total_probs[threshold + 1:])
    return {
        "low_probability_pct": round(low_prob * 100, 2),
        "high_probability_pct": round(high_prob * 100, 2),
        "threshold_low_max": threshold,
        "threshold_high_min": threshold + 1,
    }
