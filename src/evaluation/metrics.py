"""
Evaluation metrics for demand forecasting.
Includes standard ML metrics AND business-relevant KPIs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred, epsilon: float = 1.0) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def wmape(y_true, y_pred) -> float:
    """Weighted MAPE - better for retail with zero-sales weeks."""
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true) + 1) * 100)


def bias(y_true, y_pred) -> float:
    """Forecast bias - are we systematically over/under forecasting?"""
    return float(np.mean(y_pred - y_true))


def stockout_rate(y_true, y_pred, safety_stock_pct: float = 0.1) -> float:
    """% of periods where prediction underestimates demand (stockout risk)."""
    threshold = y_true * (1 - safety_stock_pct)
    return float(np.mean(y_pred < threshold) * 100)


def overstock_rate(y_true, y_pred, tolerance_pct: float = 0.2) -> float:
    """% of periods where we over-order by >20%."""
    return float(np.mean(y_pred > y_true * (1 + tolerance_pct)) * 100)


def evaluate_all(y_true, y_pred, model_name: str = "") -> dict:
    y_true = np.array(y_true)
    y_pred = np.maximum(np.array(y_pred), 0)
    metrics = {
        "model":          model_name,
        "mae":            round(mean_absolute_error(y_true, y_pred), 3),
        "rmse":           round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "mape":           round(mape(y_true, y_pred), 2),
        "wmape":          round(wmape(y_true, y_pred), 2),
        "bias":           round(bias(y_true, y_pred), 3),
        "stockout_rate":  round(stockout_rate(y_true, y_pred), 2),
        "overstock_rate": round(overstock_rate(y_true, y_pred), 2),
    }
    return metrics
