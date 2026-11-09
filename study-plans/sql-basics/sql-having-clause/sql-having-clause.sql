-- Returns: customer, total_orders, total_spent.
SELECT 
    customer, 
    COUNT(id) AS total_orders, 
    SUM(amount) AS total_spent
FROM 
    orders
GROUP BY 
    customer
HAVING 
    COUNT(id) >= 2
ORDER BY 
    total_spent DESC;
