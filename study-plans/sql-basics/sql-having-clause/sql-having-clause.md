## The HAVING Clause

HAVING filters groups after aggregation, just as WHERE filters rows before aggregation. You cannot use WHERE to filter on an aggregate result because WHERE runs before GROUP BY.

### Execution Order

```
FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY
```

WHERE filters individual rows before grouping. HAVING filters entire groups after aggregation.

### Syntax

```sql
SELECT department, COUNT(*) AS headcount
FROM employees
GROUP BY department
HAVING COUNT(*) >= 5;
```

This returns only departments with 5 or more employees. Writing WHERE COUNT(*) >= 5 would produce an error.

### WHERE vs HAVING

| Clause | Filters | Runs |
|--------|---------|------|
| WHERE | Individual rows | Before GROUP BY |
| HAVING | Grouped results | After GROUP BY |

You can use both in the same query. WHERE narrows down the rows first, then GROUP BY groups them, then HAVING filters the groups:

```sql
SELECT department, AVG(salary) AS avg_salary
FROM employees
WHERE hire_date >= '2020-01-01'
GROUP BY department
HAVING AVG(salary) > 60000;
```

### HAVING Without GROUP BY

Technically, HAVING can appear without GROUP BY - the entire table is treated as one group. This is rarely useful in practice but valid SQL.