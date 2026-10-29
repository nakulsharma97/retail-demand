"""
Synthetic Multi-Store Retail Dataset Generator
Produces realistic sales data with seasonality, promotions, and noise.
Aligned with Boots store types: pharmacy, beauty, full.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from loguru import logger

RAW_DIR = Path("data/raw")
DB_PATH = "boots_retail.db"

STORES = [
    ("S001", "Boots Nottingham City", "East Midlands", "full"),
    ("S002", "Boots Birmingham Bull Ring", "West Midlands", "full"),
    ("S003", "Boots Leeds Kirkgate", "Yorkshire", "full"),
    ("S004", "Boots London Oxford St", "London", "full"),
    ("S005", "Boots Manchester Arndale", "North West", "full"),
    ("S006", "Boots Bristol Cabot", "South West", "pharmacy"),
    ("S007", "Boots Edinburgh Princes", "Scotland", "beauty"),
    ("S008", "Boots Cardiff Queens", "Wales", "pharmacy"),
]

CATEGORIES = {
    "Skincare": ["Moisturiser SPF30", "Eye Cream", "Serum Vitamin C", "Face Wash", "Toner"],
    "Haircare": ["Shampoo 500ml", "Conditioner 500ml", "Hair Mask", "Dry Shampoo"],
    "Pharmacy": ["Paracetamol 500mg", "Ibuprofen 200mg", "Vitamin D 1000IU", "Omega-3", "Antihistamine"],
    "Beauty":   ["Foundation No7", "Mascara Volume", "Lip Gloss", "Blush Palette", "BB Cream"],
    "Baby":     ["Baby Shampoo", "Nappy Cream", "Baby Wipes 72pk", "Baby Lotion"],
}


def generate_products() -> pd.DataFrame:
    rows = []
    sku_id = 1
    for cat, items in CATEGORIES.items():
        for item in items:
            price = round(np.random.uniform(3.5, 24.99), 2)
            rows.append({
                "sku_id": f"SKU{sku_id:04d}",
                "product_name": item,
                "category": cat,
                "price": price,
                "cost": round(price * np.random.uniform(0.3, 0.55), 2),
            })
            sku_id += 1
    return pd.DataFrame(rows)


def seasonal_factor(date: pd.Timestamp, category: str) -> float:
    """Encode domain-specific seasonality."""
    month = date.month
    factors = {
        "Skincare":  [0.9,0.85,1.0,1.05,1.1,1.15,1.2,1.15,1.0,1.05,1.1,1.3],
        "Haircare":  [1.0]*12,
        "Pharmacy":  [1.4,1.3,1.1,0.9,0.85,0.8,0.8,0.85,0.9,1.0,1.2,1.5],
        "Beauty":    [1.0,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.2,1.5],
        "Baby":      [1.05]*12,
    }
    return factors.get(category, [1.0]*12)[month - 1]


def generate_sales(products: pd.DataFrame, n_days: int = 730) -> pd.DataFrame:
    logger.info(f"Generating {n_days} days of sales data across {len(STORES)} stores...")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today().date(), periods=n_days, freq="D")
    rows = []

    for store_id, store_name, region, store_type in STORES:
        store_multiplier = np.random.uniform(0.7, 1.5)  # store-level baseline
        for _, prod in products.iterrows():
            base_demand = np.random.randint(5, 40)
            for date in dates:
                s = seasonal_factor(date, prod["category"])
                trend = 1 + 0.0002 * (date - dates[0]).days  # slight upward trend
                promo = 1.0
                promo_flag = 0
                if np.random.random() < 0.05:   # 5% promo days
                    promo = np.random.uniform(1.3, 2.0)
                    promo_flag = 1
                noise = np.random.lognormal(0, 0.15)
                qty = max(0, int(base_demand * s * trend * promo * noise * store_multiplier))
                if qty == 0:
                    continue
                rows.append({
                    "store_id": store_id,
                    "sku_id": prod["sku_id"],
                    "sale_date": date.date(),
                    "quantity": qty,
                    "revenue": round(qty * prod["price"], 2),
                    "promotion_flag": promo_flag,
                })

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df):,} sales records")
    return df


def save_to_db(stores_df, products_df, sales_df):
    conn = sqlite3.connect(DB_PATH)
    # Execute schema
    with open("sql/create_tables.sql") as f:
        conn.executescript(f.read())
    stores_df.to_sql("stores", conn, if_exists="replace", index=False)
    products_df.to_sql("products", conn, if_exists="replace", index=False)
    sales_df.to_sql("sales", conn, if_exists="append", index=False)
    conn.close()
    logger.info(f"Data saved to {DB_PATH}")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    stores_df = pd.DataFrame(STORES, columns=["store_id","store_name","region","store_type"])
    stores_df["opened_date"] = "2015-01-01"

    products_df = generate_products()
    sales_df    = generate_sales(products_df)

    stores_df.to_csv(RAW_DIR / "stores.csv", index=False)
    products_df.to_csv(RAW_DIR / "products.csv", index=False)
    sales_df.to_csv(RAW_DIR / "sales.csv", index=False)

    save_to_db(stores_df, products_df, sales_df)
    logger.success("Data generation complete.")
