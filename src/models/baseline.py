"""
Naive / baseline forecasting models.
Always benchmark against these before claiming ML improvement.
"""

import pandas as pd
import numpy as np


class NaiveLastValueModel:
    """Predict next week = last week's sales."""
    def fit(self, df): self.df = df; return self
    def predict(self, df):
        return df.groupby(["store_id","sku_id"])["total_qty"].shift(1).fillna(0)


class MovingAverageModel:
    def __init__(self, window: int = 4):
        self.window = window

    def fit(self, df):
        self.df = df
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return (
            df.sort_values(["store_id","sku_id","week_start"])
            .groupby(["store_id","sku_id"])["total_qty"]
            .transform(lambda x: x.shift(1).rolling(self.window, min_periods=1).mean())
        )


class SeasonalNaiveModel:
    """Predict next week = same week last year."""
    def fit(self, df): return self
    def predict(self, df):
        return df.groupby(["store_id","sku_id"])["total_qty"].shift(52).fillna(0)
