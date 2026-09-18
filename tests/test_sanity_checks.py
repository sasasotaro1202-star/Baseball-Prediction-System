from research.sanity_checks import target_permutation_check, validate_probability_rows

def test_probability_rows_are_validated():
    validate_probability_rows([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])

def test_probability_rows_fail_closed_on_bad_sum():
    import pytest
    with pytest.raises(ValueError):
        validate_probability_rows([[0.2, 0.2, 0.2]])

def test_permutation_check_is_deterministic_and_rejects_target_signal():
    def predict(y):
        return [[1.0, 0.0] if int(v) == 0 else [0.0, 1.0] for v in y]
    def accuracy(y, p):
        return sum(int(max(range(2), key=lambda i: row[i])) == int(v) for v, row in zip(y, p)) / len(y)
    result1 = target_permutation_check([0, 1] * 20, predict, accuracy, n_permutations=25, seed=7)
    result2 = target_permutation_check([0, 1] * 20, predict, accuracy, n_permutations=25, seed=7)
    assert result1 == result2
    assert result1.observed == 1.0
    assert result1.suspicious is True
