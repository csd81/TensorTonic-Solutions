-- Returns: product, revenue, sale_date.
SELECT product, revenue, sale_date
FROM sales
ORDER BY 
    revenue DESC NULLS LAST, 
    sale_date ASC NULLS LAST
LIMIT 3 
OFFSET 1;
