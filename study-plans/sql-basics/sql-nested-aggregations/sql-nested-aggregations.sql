-- Returns: avg_daily_orders, avg_daily_revenue, busiest_day_orders.
WITH daily_summaries AS (
    SELECT 
        order_date,
        COUNT(id) AS daily_order_count,
        SUM(amount) AS daily_revenue
    FROM 
        orders
    GROUP BY 
        order_date
)
SELECT 
    ROUND(AVG(daily_order_count), 2) AS avg_daily_orders,
    ROUND(AVG(daily_revenue), 2) AS avg_daily_revenue,
    MAX(daily_order_count) AS busiest_day_orders
FROM 
    daily_summaries;
