import pandas as pd

import research.mlb_candidate_replay as mlb_candidate
import research.npb_candidate_replay as npb_candidate


class _FakeScoreBT:
    def fit_score_ensemble(self, *args, **kwargs):
        return object()

    def predict_scores(self, *args, **kwargs):
        return 2.0, 1.8, 0.25


def _games():
    return pd.DataFrame([
        {"home_score": 1, "away_score": 1, "datetime": "2026-06-01T00:00:00Z"}
    ])


def _x():
    return pd.DataFrame([{"dummy": 1.0}])


def test_npb_holdout_uses_shared_lambda(monkeypatch):
    seen = {}

    monkeypatch.setattr(npb_candidate, "score_candidates",
                        lambda h, a, shared: (seen.update(shared=shared) or [("1-1", 1.0)]))
    monkeypatch.setattr(npb_candidate, "low_high_probs",
                        lambda h, a, shared: (seen.update(low_high_shared=shared) or (0.5, 0.5)))

    score, hilo = npb_candidate._target_metrics(
        _FakeScoreBT(), _x(), _games(), _games(), _x(), pd.DataFrame([[0.6, 0.1, 0.3]]).to_numpy(),
    )

    assert seen["shared"] == 0.25
    assert seen["low_high_shared"] == 0.25
    assert score["rows"] == 1
    assert hilo["rows"] == 1


def test_mlb_holdout_uses_shared_lambda(monkeypatch):
    seen = {}

    monkeypatch.setattr(mlb_candidate, "score_candidates",
                        lambda h, a, shared: (seen.update(shared=shared) or [("1-1", 1.0)]))
    monkeypatch.setattr(mlb_candidate, "low_high_probs",
                        lambda h, a, shared: (seen.update(low_high_shared=shared) or (0.5, 0.5)))

    score, hilo = mlb_candidate._target_metrics(
        _FakeScoreBT(), _x(), _games(), _games(), _x(), pd.DataFrame([[0.6, 0.4]]).to_numpy(),
        object(),
    )

    assert seen["shared"] == 0.25
    assert seen["low_high_shared"] == 0.25
    assert score["rows"] == 1
    assert hilo["rows"] == 1
