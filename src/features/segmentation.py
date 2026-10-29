"""
Customer Segmentation using RFM Analysis + KMeans Clustering.
RFM (Recency, Frequency, Monetary) is a classic retail analytics technique.
Results exported as a customer_segments.csv for Power BI / Streamlit visualisation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path


def compute_rfm(sales_df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """Compute RFM metrics. Uses store_id as proxy customer (for SKU-level data)."""
    if snapshot_date is None:
        snapshot_date = sales_df["sale_date"].max() + pd.Timedelta(days=1)

    rfm = (
        sales_df.groupby("store_id")
        .agg(
            recency=("sale_date", lambda x: (snapshot_date - x.max()).days),
            frequency=("sale_date", "count"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )
    return rfm


def rfm_score(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign scores to each RFM dimension using quantile binning."""
    n = len(rfm)
    q = min(5, n)  # use fewer bins if not enough data points

    rfm["R_score"] = pd.qcut(rfm["recency"],   q=q, labels=False, duplicates="drop")
    rfm["F_score"] = pd.qcut(rfm["frequency"], q=q, labels=False, duplicates="drop")
    rfm["M_score"] = pd.qcut(rfm["monetary"],  q=q, labels=False, duplicates="drop")

    # Reverse recency so lower recency (more recent) = higher score
    rfm["R_score"] = rfm["R_score"].max() - rfm["R_score"]

    rfm[["R_score", "F_score", "M_score"]] = (
        rfm[["R_score", "F_score", "M_score"]].fillna(0).astype(int)
    )
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    return rfm


def kmeans_segmentation(rfm: pd.DataFrame, n_clusters: int = 4) -> tuple:
    n_clusters = min(n_clusters, len(rfm))  # never more clusters than data points

    features = rfm[["recency", "frequency", "monetary"]].copy()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(X_scaled)

    if n_clusters > 1:
        sil = silhouette_score(X_scaled, rfm["cluster"])
        logger.info(f"KMeans silhouette score: {sil:.3f}")

    # Map cluster IDs to business-friendly labels
    cluster_labels = {0: "Champions", 1: "Loyal", 2: "At Risk", 3: "Lost"}
    rfm["segment"] = rfm["cluster"].map(cluster_labels).fillna("Other")

    return rfm, km, scaler


if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv("data/raw/sales.csv", parse_dates=["sale_date"])
    rfm   = compute_rfm(sales)
    rfm   = rfm_score(rfm)
    rfm, km, scaler = kmeans_segmentation(rfm)
    rfm.to_csv("data/processed/customer_segments.csv", index=False)
    print(rfm[["store_id", "segment", "RFM_score"]].sort_values("RFM_score", ascending=False))
    logger.success("Segmentation complete.")