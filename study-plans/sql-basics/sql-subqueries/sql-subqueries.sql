-- Returns: name, price, vs_avg.
SELECT 
    name,
    price,
    ROUND(price - (SELECT AVG(price) FROM products), 2) AS vs_avg
FROM products
WHERE id IN (
    SELECT DISTINCT product_id 
    FROM sales
)
ORDER BY vs_avg DESC, name ASC;
