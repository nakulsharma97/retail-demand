-- ============================================================
-- Boots Retail Demand Forecasting - Database Schema
-- Compatible with SQLite, PostgreSQL, Databricks SQL
-- ============================================================

CREATE TABLE IF NOT EXISTS stores (
    store_id        TEXT PRIMARY KEY,
    store_name      TEXT NOT NULL,
    region          TEXT NOT NULL,
    store_type      TEXT CHECK(store_type IN ('pharmacy','beauty','full')),
    opened_date     DATE
);

CREATE TABLE IF NOT EXISTS products (
    sku_id          TEXT PRIMARY KEY,
    product_name    TEXT NOT NULL,
    category        TEXT NOT NULL,
    subcategory     TEXT,
    price           REAL NOT NULL,
    cost            REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sales (
    sale_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id        TEXT REFERENCES stores(store_id),
    sku_id          TEXT REFERENCES products(sku_id),
    sale_date       DATE NOT NULL,
    quantity        INTEGER NOT NULL,
    revenue         REAL NOT NULL,
    promotion_flag  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id     TEXT PRIMARY KEY,
    store_id        TEXT REFERENCES stores(store_id),
    join_date       DATE,
    age_band        TEXT,
    loyalty_tier    TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id        TEXT,
    sku_id          TEXT,
    forecast_date   DATE,
    model_name      TEXT,
    predicted_qty   REAL,
    lower_bound     REAL,
    upper_bound     REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for analytical query performance
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_sales_store ON sales(store_id);
CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales(sku_id);
