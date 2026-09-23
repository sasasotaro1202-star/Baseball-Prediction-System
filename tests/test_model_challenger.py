from baseball_backtest import BaseballBacktest
from pathlib import Path


def test_knn_analog_challenger_is_registered_and_pipeline_backed():
    bt = BaseballBacktest(Path("data"))
    models = bt.models("NPB")
    assert "KNNAnalog" in models
    model = models["KNNAnalog"]
    assert hasattr(model, "named_steps")
    assert "scale" in model.named_steps
    assert "m" in model.named_steps
    assert model.named_steps["m"].n_neighbors == 45
