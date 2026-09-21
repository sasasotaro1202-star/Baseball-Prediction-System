import pandas as pd

from research.closed_loop_execute import hilo_probs


def test_hilo_uses_total_run_boundary_not_per_team_thresholds():
    df = pd.DataFrame([{"lambda_home": 2.0, "lambda_away": 2.0}])
    low, high = hilo_probs(df)[0]
    # P(H+A <= 6) for independent Poisson(2)+Poisson(2) = Poisson(4).
    expected_low = sum(__import__("math").exp(-4.0) * 4.0**k / __import__("math").factorial(k) for k in range(7))
    assert abs(float(low) - expected_low) < 1e-10
    assert abs(float(low) + float(high) - 1.0) < 1e-12
