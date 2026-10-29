-- ============================================================
-- Business KPI Aggregation Queries
-- ============================================================

-- Top 10 products by revenue
SELECT
    p.product_name,
    p.category,
    SUM(s.revenue) AS total_revenue,
    SUM(s.quantity) AS total_units
FROM sales s
JOIN products p ON s.sku_id = p.sku_id
GROUP BY p.product_name, p.category
ORDER BY total_revenue DESC
LIMIT 10;

-- Store performance comparison
SELECT
    st.store_name,
    st.region,
    COUNT(DISTINCT s.sku_id)    AS unique_skus_sold,
    SUM(s.revenue)              AS total_revenue,
    SUM(s.quantity)             AS total_units,
    AVG(s.revenue / s.quantity) AS avg_unit_price
FROM sales s
JOIN stores st ON s.store_id = st.store_id
GROUP BY st.store_name, st.region
ORDER BY total_revenue DESC;

-- Promotion uplift analysis
SELECT
    promotion_flag,
    AVG(quantity)   AS avg_qty,
    SUM(quantity)   AS total_qty,
    COUNT(*)        AS num_transactions
FROM sales
GROUP BY promotion_flag;
