from baseball_backtest import BaseballBacktest

def test_fast_model_pool_is_explicit_and_bounded(monkeypatch):
    monkeypatch.setenv("BASEBALL_FAST_OOS", "1")
    monkeypatch.setenv(
        "BASEBALL_FAST_MODEL_POOL",
        "Logistic,HistGB,ExtraTrees,HierarchicalDrawResult",
    )
    bt = BaseballBacktest()
    mlb = list(bt.models("MLB"))
    npb = list(bt.models("NPB"))
    assert mlb == ["Logistic", "HistGB", "ExtraTrees"]
    assert npb == [
        "Logistic",
        "HistGB",
        "ExtraTrees",
        "HierarchicalDrawResult",
    ]


def test_fast_model_pool_rejects_unknown_model(monkeypatch):
    monkeypatch.setenv("BASEBALL_FAST_OOS", "1")
    monkeypatch.setenv("BASEBALL_FAST_MODEL_POOL", "Logistic,NotARealModel")
    bt = BaseballBacktest()
    try:
        bt.models("MLB")
    except ValueError as exc:
        assert "NotARealModel" in str(exc)
    else:
        raise AssertionError("unknown fast model must fail closed")
