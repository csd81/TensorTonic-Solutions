## Nested Aggregations

Sometimes you need to aggregate data that is itself the result of an aggregation. For example, finding the average of daily totals or the maximum of monthly counts. SQL does not allow nesting aggregate functions directly - AVG(SUM(amount)) is a syntax error. You must use a subquery or CTE.

### The Two-Step Pattern

Step 1: Compute the inner aggregation in a subquery or CTE.
Step 2: Aggregate the results of step 1 in the outer query.

```sql
-- Average of daily totals
SELECT ROUND(AVG(daily_total), 2) AS avg_daily_total
FROM (
    SELECT order_date, SUM(amount) AS daily_total
    FROM orders
    GROUP BY order_date
) daily_sums;
```

### Using a CTE

The same logic using WITH:

```sql
WITH daily_sums AS (
    SELECT order_date, SUM(amount) AS daily_total
    FROM orders
    GROUP BY order_date
)
SELECT ROUND(AVG(daily_total), 2) AS avg_daily_total
FROM daily_sums;
```

CTEs are often easier to read, especially when the inner query is complex.

### Why Not Nest Directly?

SQL processes GROUP BY once per query level. When you write SELECT AVG(SUM(x)), the engine cannot determine which GROUP BY applies to SUM and which to AVG. The subquery approach makes the execution order explicit: the inner query groups and aggregates first, then the outer query aggregates those results.

### Common Use Cases

- Average of per-group totals (e.g., average revenue per store)
- Max/min of per-group counts (e.g., busiest day by order count)
- Standard deviation of per-category averages