## LEFT JOIN

A LEFT JOIN returns all rows from the left table, even if there is no matching row in the right table. Unmatched right-side columns are filled with NULL.

```sql
SELECT c.name, o.order_id
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;
```

### LEFT JOIN vs INNER JOIN

- INNER JOIN: only rows with matches in both tables
- LEFT JOIN: all rows from the left table, NULLs for unmatched right columns

The "left" table is the one in the FROM clause (or the one that appears before the JOIN keyword).

### Finding Orphaned Records

A powerful pattern is combining LEFT JOIN with an IS NULL check to find rows in the left table that have no match in the right table:

```sql
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;
```

This returns customers who have never placed an order. The LEFT JOIN keeps them in the result with NULL order columns, and the WHERE filter keeps only those NULL rows.

### Multiple LEFT JOINs

You can chain multiple LEFT JOINs. Each preserves all rows from the accumulated result so far:

```sql
SELECT e.name, d.dept_name, m.name AS manager
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id
LEFT JOIN employees m ON e.manager_id = m.id;
```

### NULL Behavior

Remember that NULLs from unmatched joins propagate through expressions. If you compute right_table.price * 2, the result is NULL when there is no match. Use COALESCE to provide defaults:

```sql
SELECT c.name, COALESCE(o.total, 0) AS total
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;
```
