## Conditional Aggregation with CASE

Conditional aggregation combines CASE expressions with aggregate functions to compute multiple filtered aggregates in a single pass over the data. This lets you pivot rows into columns without a separate PIVOT clause.

### The Pattern

```sql
SELECT
    category,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_count
FROM orders
GROUP BY category;
```

Each CASE expression evaluates per row, producing 1 or 0. SUM then totals these per group.

### COUNT vs SUM Approach

Two equivalent patterns exist:

```sql
-- SUM approach: explicit 1/0
SUM(CASE WHEN condition THEN 1 ELSE 0 END)

-- COUNT approach: NULL trick
COUNT(CASE WHEN condition THEN 1 END)
```

COUNT ignores NULLs, so when the CASE has no ELSE (defaulting to NULL), non-matching rows are skipped. Both produce the same result.

### Aggregating Values Conditionally

You can also sum actual values, not just counts:

```sql
SELECT
    department,
    SUM(CASE WHEN quarter = 'Q1' THEN revenue ELSE 0 END) AS q1_revenue,
    SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END) AS q2_revenue
FROM sales
GROUP BY department;
```

### Why Not Multiple Queries?

Without conditional aggregation, you would need separate queries (or subqueries) for each condition, then join the results. Conditional aggregation does it in one scan of the table, which is both simpler and faster.