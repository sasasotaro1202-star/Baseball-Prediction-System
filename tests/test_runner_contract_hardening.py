import pytest
from prediction.runner import _validate_final_probabilities, _validate_score_candidates


def test_probability_mapping_rejects_nan_and_inf():
    with pytest.raises(ValueError):
        _validate_final_probabilities({"home": float("nan"), "away": 1.0}, "MLB")
    with pytest.raises(ValueError):
        _validate_final_probabilities({"home": float("inf"), "away": 0.0}, "MLB")


def test_probability_contract_rejects_unknown_keys():
    with pytest.raises(ValueError):
        _validate_final_probabilities({"home": 0.5, "away": 0.4, "draw": 0.1}, "MLB")


def test_score_candidates_reject_invalid_probability():
    with pytest.raises(ValueError):
        _validate_score_candidates([{"score": "3-2", "probability": float("nan")}])


def test_score_candidates_reject_malformed_item():
    with pytest.raises(ValueError):
        _validate_score_candidates([{"score": "3-2"}])
