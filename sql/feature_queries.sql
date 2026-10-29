-- ============================================================
-- Feature Engineering Queries
-- ============================================================

-- Weekly sales aggregation per store-SKU
CREATE VIEW IF NOT EXISTS weekly_sales AS
SELECT
    store_id,
    sku_id,
    strftime('%Y-%W', sale_date)    AS year_week,
    SUM(quantity)                   AS total_qty,
    SUM(revenue)                    AS total_revenue,
    AVG(quantity)                   AS avg_daily_qty,
    MAX(promotion_flag)             AS had_promotion
FROM sales
GROUP BY store_id, sku_id, year_week;

-- 4-week rolling average per store-SKU
SELECT
    store_id,
    sku_id,
    year_week,
    total_qty,
    AVG(total_qty) OVER (
        PARTITION BY store_id, sku_id
        ORDER BY year_week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS rolling_4w_avg
FROM weekly_sales;
