-- Returns: month, revenue, prev_revenue, revenue_change.
SELECT 
    month,
    revenue,
    COALESCE(LAG(revenue) OVER (ORDER BY month ASC, id ASC), 0) AS prev_revenue,
    revenue - COALESCE(LAG(revenue) OVER (ORDER BY month ASC, id ASC), 0) AS revenue_change
FROM 
    monthly_revenue
ORDER BY 
    month ASC, 
    id ASC;
