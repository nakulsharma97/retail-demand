"""
Data preprocessing and cleaning pipeline.
Handles missing values, type casting, and train/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/processed")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stores   = pd.read_csv("data/raw/stores.csv")
    products = pd.read_csv("data/raw/products.csv")
    sales    = pd.read_csv("data/raw/sales.csv", parse_dates=["sale_date"])
    return stores, products, sales


def build_master_df(sales, stores, products) -> pd.DataFrame:
    df = (
        sales
        .merge(stores[["store_id","region","store_type"]], on="store_id")
        .merge(products[["sku_id","category","price"]], on="sku_id")
    )
    df = df.sort_values(["store_id","sku_id","sale_date"]).reset_index(drop=True)
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to weekly store-SKU level for modelling."""
    df["year_week"] = df["sale_date"].dt.to_period("W")
    weekly = (
        df.groupby(["store_id","sku_id","category","region","year_week"])
        .agg(
            total_qty=("quantity","sum"),
            total_revenue=("revenue","sum"),
            had_promotion=("promotion_flag","max"),
        )
        .reset_index()
    )
    weekly["week_start"] = weekly["year_week"].dt.start_time
    return weekly.sort_values(["store_id","sku_id","week_start"])


def train_test_split_temporal(df: pd.DataFrame, test_weeks: int = 8):
    cutoff = df["week_start"].max() - pd.Timedelta(weeks=test_weeks)
    train  = df[df["week_start"] <= cutoff]
    test   = df[df["week_start"] >  cutoff]
    logger.info(f"Train: {len(train):,} rows | Test: {len(test):,} rows | Cutoff: {cutoff.date()}")
    return train, test


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stores, products, sales = load_raw_data()
    master = build_master_df(sales, stores, products)
    weekly = aggregate_weekly(master)
    train, test = train_test_split_temporal(weekly)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR  / "test.csv",  index=False)
    weekly.to_csv(PROCESSED_DIR / "weekly_master.csv", index=False)
    logger.success("Preprocessing complete.")
