## Subqueries

A subquery is a query nested inside another query. It can appear in the SELECT list, the FROM clause, or the WHERE clause. Subqueries let you compute intermediate results without creating temporary tables.

### Subquery in WHERE

Filter rows based on a computed set or value:

```sql
SELECT name FROM products
WHERE id IN (SELECT product_id FROM sales);
```

The inner query returns a list of product IDs that have sales. The outer query filters to only those products.

### Subquery in SELECT (Scalar Subquery)

Return a single value to use as a column:

```sql
SELECT name, price,
       price - (SELECT AVG(price) FROM products) AS vs_avg
FROM products;
```

The subquery computes one number (the average price), and each row subtracts it from its own price.

### Subquery in FROM (Derived Table)

Use a subquery as a virtual table:

```sql
SELECT d.category, d.total
FROM (SELECT category, SUM(amount) AS total
      FROM orders GROUP BY category) AS d
WHERE d.total > 1000;
```

### Correlated vs. Uncorrelated

An **uncorrelated** subquery runs once and returns the same result for every row. It does not reference the outer query:

```sql
WHERE price > (SELECT AVG(price) FROM products)
```

A **correlated** subquery references a column from the outer query and runs once per outer row:

```sql
WHERE EXISTS (SELECT 1 FROM sales WHERE sales.product_id = products.id)
```

Correlated subqueries are more flexible but slower on large tables because they execute repeatedly.

### EXISTS vs. IN

Both filter based on another query, but they differ:

- IN: checks if a value is in a list. The subquery returns a column of values.
- EXISTS: checks if the subquery returns any rows at all. Often faster for large datasets because it can stop at the first match.
