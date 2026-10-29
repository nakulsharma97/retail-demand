import pytest
import numpy as np
from src.evaluation.metrics import mape, wmape, bias, evaluate_all

def test_mape_perfect():
    assert mape([10,20,30], [10,20,30]) == pytest.approx(0.0, abs=0.01)

def test_mape_nonzero():
    result = mape([100,200], [90,180])
    assert result > 0

def test_evaluate_all_keys():
    result = evaluate_all([10,20,30], [9,21,29], "test_model")
    assert "mae" in result
    assert "rmse" in result
    assert "mape" in result
    assert "stockout_rate" in result
