import numpy as np
import pytest

from evaluation.calibration import TemperatureCalibration, fit_temperature


def test_calibration_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        TemperatureCalibration().transform(np.empty((0, 3)))


def test_calibration_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="finite"):
        TemperatureCalibration().transform(np.array([[0.5, np.nan]]))


def test_calibration_normalizes_rows():
    out = TemperatureCalibration(temperature=1.0).transform(np.array([[2.0, 1.0]]))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all(out >= 0)


def test_fit_temperature_rejects_invalid_grid():
    with pytest.raises(ValueError, match="grid"):
        fit_temperature(np.array([[0.6, 0.4]]), np.array([0]), grid=np.array([0.0, np.nan]))
