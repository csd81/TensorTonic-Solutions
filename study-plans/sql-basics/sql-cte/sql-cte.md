## Common Table Expressions (CTEs)

A Common Table Expression (CTE) is a named temporary result set defined at the beginning of a query using the WITH keyword. It exists only for the duration of that single query and makes complex queries easier to read and maintain.

### Basic WITH Syntax

```sql
WITH cte_name AS (
    SELECT column1, column2
    FROM some_table
    WHERE condition
)
SELECT *
FROM cte_name
WHERE another_condition;
```

The CTE is defined inside parentheses after AS. The main query that follows can reference cte_name as if it were a regular table.

### Naming CTEs

CTE names follow the same rules as table names. Choose descriptive names that communicate what the intermediate result represents:

```sql
WITH monthly_totals AS (
    SELECT EXTRACT(MONTH FROM order_date) AS month,
           SUM(amount) AS total
    FROM orders
    GROUP BY EXTRACT(MONTH FROM order_date)
)
SELECT month, total
FROM monthly_totals
WHERE total > 10000;
```

### Multiple CTEs

You can define several CTEs separated by commas. Later CTEs can reference earlier ones:

```sql
WITH regional_sales AS (
    SELECT region, SUM(amount) AS total_sales
    FROM orders
    GROUP BY region
),
top_regions AS (
    SELECT region
    FROM regional_sales
    WHERE total_sales > (SELECT SUM(total_sales) / 10 FROM regional_sales)
)
SELECT region, product, SUM(quantity) AS units
FROM orders
WHERE region IN (SELECT region FROM top_regions)
GROUP BY region, product;
```

### CTE vs. Subquery

CTEs and subqueries can often achieve the same result, but they differ in readability and reuse:

- **Readability**: CTEs separate the logic into named steps. A deeply nested subquery can be hard to follow, while a CTE reads top-to-bottom.
- **Reuse**: A CTE can be referenced multiple times in the main query. A subquery would need to be duplicated each time.
- **Performance**: In most databases, CTEs and equivalent subqueries produce the same execution plan. Some engines materialize CTEs (compute them once and store the result), while others inline them like subqueries.

Use a CTE when:
- The intermediate result is referenced more than once.
- The query has multiple logical steps that benefit from being named.
- You want to make the query easier for others (or your future self) to understand.

Use a subquery when:
- The intermediate result is simple and used only once.
- You want to keep things compact without defining a named block.

### CTE with Aggregation and Filtering

A common pattern is to aggregate data in a CTE and then filter the aggregated results in the main query:

```sql
WITH customer_stats AS (
    SELECT customer_id, COUNT(*) AS num_orders, AVG(amount) AS avg_order
    FROM orders
    GROUP BY customer_id
)
SELECT customer_id, num_orders, avg_order
FROM customer_stats
WHERE num_orders >= 5;
```

This is equivalent to using HAVING directly, but the CTE version is clearer when the filtering logic is complex or when you need the aggregated data for additional joins.
