## GROUP BY Fundamentals

GROUP BY collapses multiple rows that share a common value into a single summary row. It is the foundation of aggregation in SQL.

### Syntax

```sql
SELECT column, AGG_FUNCTION(other_column)
FROM table
GROUP BY column;
```

### The Golden Rule

Every column in the SELECT list must either appear in the GROUP BY clause or be wrapped in an aggregate function. If you SELECT customer and SUM(amount), then customer must be in GROUP BY. Violating this rule produces an error in most databases.

### Common Aggregate Functions

| Function | Purpose |
|----------|---------|
| COUNT(*) | Number of rows in the group |
| COUNT(col) | Number of non-NULL values in col |
| SUM(col) | Total of all values in col |
| AVG(col) | Average of all values in col |
| MIN(col) | Smallest value in col |
| MAX(col) | Largest value in col |

### How GROUP BY Executes

1. The database scans the table (applying any WHERE filter first).
2. Rows are partitioned into groups based on the GROUP BY column(s).
3. Each aggregate function runs once per group, producing one output row per group.

### Ordering Groups

GROUP BY does not guarantee any particular order. To sort the results, always add an explicit ORDER BY clause - you can order by the aggregate column or the grouped column.

### Multiple GROUP BY Columns

You can group by more than one column:

```sql
SELECT department, job_title, COUNT(*)
FROM employees
GROUP BY department, job_title;
```

This creates one group for each unique (department, job_title) combination.