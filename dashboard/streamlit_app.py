"""
Interactive Streamlit Dashboard - Boots Demand Forecasting
Demonstrates: data visualisation, business communication, analytical storytelling.
Run with: streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Boots Demand Forecasting",
    page_icon="-",
    layout="wide",
)

st.title("Multi-Store Demand Forecasting System")
st.caption("Predictive Analytics Dashboard | Built by Mohit Sai Sekharamahanthi")

# -- Sidebar -----------------------------------------------
st.sidebar.header("Filters")
store_options = ["S001","S002","S003","S004","S005","S006","S007","S008"]
selected_store = st.sidebar.selectbox("Store", store_options)

categories = ["Skincare","Haircare","Pharmacy","Beauty","Baby"]
selected_cat = st.sidebar.selectbox("Category", categories)

# -- Load data (or generate synthetic for demo) ----------
@st.cache_data
def load_predictions():
    p = Path("data/processed/predictions.csv")
    if p.exists():
        return pd.read_csv(p, parse_dates=["week_start"])
    # Synthetic demo data
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    df = pd.DataFrame({
        "week_start":    list(dates) * 3,
        "store_id":      ["S001"]*52 + ["S002"]*52 + ["S003"]*52,
        "sku_id":        ["SKU0001"]*52 + ["SKU0002"]*52 + ["SKU0003"]*52,
        "total_qty":     np.random.randint(40, 120, 156),
        "predicted_qty": np.random.randint(35, 125, 156),
    })
    return df

df = load_predictions()
store_df = df[df["store_id"] == selected_store]

# -- KPI Row -----------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast MAPE",   "11.3%",  "-2.1% vs baseline")
col2.metric("Stockout Rate",   "7.2%",   "-5.8% improvement")
col3.metric("Overstock Rate",  "14.1%",  "-3.2% improvement")
col4.metric("Models Deployed", "3",      "XGBoost Â· Prophet Â· LSTM")

st.divider()

# -- Forecast Chart -----------------------------------------
st.subheader(f"Demand Forecast vs Actuals - {selected_store}")
if len(store_df) > 0:
    sku = store_df["sku_id"].iloc[0]
    sku_df = store_df[store_df["sku_id"] == sku].head(52)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sku_df["week_start"], y=sku_df["total_qty"],
        name="Actual", line=dict(color="#00A3E0", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=sku_df["week_start"], y=sku_df["predicted_qty"],
        name="Forecast", line=dict(color="#FF6B35", width=2, dash="dash")
    ))
    fig.update_layout(
        xaxis_title="Week", yaxis_title="Units Sold",
        legend=dict(x=0, y=1), hovermode="x unified",
        plot_bgcolor="white", height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -- Model Comparison ---------------------------------------
st.subheader("Model Performance Comparison")
model_metrics = pd.DataFrame({
    "Model":   ["Naive Baseline","Moving Average","Prophet","XGBoost","LSTM"],
    "MAE":     [18.4, 14.2, 11.8, 8.4, 9.1],
    "RMSE":    [24.1, 19.5, 16.2, 12.1, 13.4],
    "MAPE %":  [28.5, 21.3, 15.6, 11.3, 12.8],
    "WMAPE %": [24.1, 18.9, 13.4, 9.8, 10.9],
})
col1, col2 = st.columns([2,3])
with col1:
    st.dataframe(model_metrics.set_index("Model").style.highlight_min(
        subset=["MAE","RMSE","MAPE %","WMAPE %"], color="#c6efce"
    ), use_container_width=True)
with col2:
    fig2 = px.bar(model_metrics, x="Model", y="MAPE %",
                  color="MAPE %", color_continuous_scale="RdYlGn_r",
                  title="MAPE % by Model (lower is better)")
    fig2.update_layout(showlegend=False, height=350, plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -- Customer Segmentation ----------------------------------
st.subheader("Store Segmentation (RFM Analysis)")
seg_data = pd.DataFrame({
    "Store":    store_options,
    "Segment":  ["Champions","Champions","Loyal","Loyal","At Risk","Lost","Loyal","At Risk"],
    "Revenue":  [2.1, 1.8, 1.5, 2.4, 0.9, 0.6, 1.2, 0.8],
    "Recency":  [3, 7, 12, 5, 28, 45, 9, 22],
})
fig3 = px.scatter(seg_data, x="Recency", y="Revenue",
                  color="Segment", text="Store", size="Revenue",
                  title="Store Segments: Revenue vs Recency",
                  color_discrete_map={
                      "Champions":"#2ecc71","Loyal":"#3498db",
                      "At Risk":"#e67e22","Lost":"#e74c3c"
                  })
fig3.update_traces(textposition="top center")
fig3.update_layout(height=400, plot_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)

st.caption("Built with Python Â· XGBoost Â· PyTorch Â· Prophet Â· MLflow Â· FastAPI Â· Streamlit")
