"""
Time-series feature engineering for demand forecasting.
Creates lag features, rolling statistics, calendar features.
These are the core predictors used by XGBoost and LSTM models.
"""

import pandas as pd
import numpy as np
from loguru import logger


UK_BANK_HOLIDAYS_2023_2024 = pd.to_datetime([
    "2023-01-02","2023-04-07","2023-04-10","2023-05-01",
    "2023-05-08","2023-05-29","2023-08-28","2023-12-25","2023-12-26",
    "2024-01-01","2024-03-29","2024-04-01","2024-05-06",
    "2024-05-27","2024-08-26","2024-12-25","2024-12-26",
])


def add_calendar_features(df: pd.DataFrame, date_col: str = "week_start") -> pd.DataFrame:
    df = df.copy()
    df["week_of_year"]  = df[date_col].dt.isocalendar().week.astype(int)
    df["month"]         = df[date_col].dt.month
    df["quarter"]       = df[date_col].dt.quarter
    df["year"]          = df[date_col].dt.year
    df["is_december"]   = (df["month"] == 12).astype(int)
    df["is_q4"]         = (df["quarter"] == 4).astype(int)
    # Proximity to bank holidays
    df["days_to_holiday"] = df[date_col].apply(
        lambda d: min((abs((d - h).days) for h in UK_BANK_HOLIDAYS_2023_2024), default=365)
    )
    df["near_holiday"]  = (df["days_to_holiday"] <= 7).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "total_qty",
                     lags: list = [1, 2, 4, 8, 13, 26, 52]) -> pd.DataFrame:
    df = df.sort_values(["store_id","sku_id","week_start"])
    grp = df.groupby(["store_id","sku_id"])[target]
    for lag in lags:
        df[f"lag_{lag}w"] = grp.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "total_qty") -> pd.DataFrame:
    df = df.sort_values(["store_id","sku_id","week_start"])
    grp = df.groupby(["store_id","sku_id"])[target]
    for window in [4, 8, 13]:
        df[f"roll_mean_{window}w"] = grp.shift(1).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"roll_std_{window}w"]  = grp.shift(1).transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
    # Exponentially weighted mean (captures recent trend)
    df["ewm_mean_4w"] = grp.shift(1).transform(lambda x: x.ewm(span=4).mean())
    return df


def add_category_encoding(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["category","store_type","region"] if c in df.columns]
    if cols:
        df = pd.get_dummies(df, columns=cols, drop_first=False)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building feature matrix...")
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_category_encoding(df)
    # Drop rows where lag features are NaN (first N weeks per group)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    logger.info(f"Feature matrix shape: {df.shape}")
    return df


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv", parse_dates=["week_start"])
    test  = pd.read_csv("data/processed/test.csv",  parse_dates=["week_start"])
    combined = pd.concat([train, test]).sort_values(["store_id","sku_id","week_start"])
    featured = build_feature_matrix(combined)
    featured.to_csv("data/processed/featured.csv", index=False)
    print("Done. Shape:", featured.shape)
