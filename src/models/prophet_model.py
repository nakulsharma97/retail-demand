"""
Facebook Prophet Demand Forecasting Model
Best for explainability and handling seasonality + holidays.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
from loguru import logger


UK_HOLIDAYS = pd.DataFrame({
    "holiday": "uk_bank_holiday",
    "ds": pd.to_datetime([
        "2023-01-02","2023-04-07","2023-04-10","2023-05-01",
        "2023-05-08","2023-05-29","2023-08-28","2023-12-25","2023-12-26",
        "2024-01-01","2024-03-29","2024-04-01","2024-05-06",
        "2024-05-27","2024-08-26","2024-12-25","2024-12-26",
    ]),
    "lower_window": -2,
    "upper_window": 1,
})


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"week_start": "ds", "total_qty": "y"})[["ds","y"]]


def train_prophet(train_df: pd.DataFrame, store_id: str, sku_id: str) -> Prophet:
    prophet_df = prepare_prophet_df(train_df)
    model = Prophet(
        holidays=UK_HOLIDAYS,
        yearly_seasonality=True,
        weekly_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
    )
    model.add_regressor("had_promotion") if "had_promotion" in train_df.columns else None

    with mlflow.start_run(run_name=f"prophet_{store_id}_{sku_id}"):
        model.fit(prophet_df)
        future   = model.make_future_dataframe(periods=8, freq="W")
        forecast = model.predict(future)
        mlflow.log_param("store_id", store_id)
        mlflow.log_param("sku_id",   sku_id)

    return model, forecast


def evaluate_prophet(model: Prophet, test_df: pd.DataFrame) -> dict:
    prophet_test = prepare_prophet_df(test_df)
    forecast     = model.predict(prophet_test)
    y_true = prophet_test["y"].values
    y_pred = forecast["yhat"].values[:len(y_true)]
    y_pred = np.maximum(y_pred, 0)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    logger.info(f"Prophet | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}
