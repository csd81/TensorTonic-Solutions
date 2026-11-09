## LIMIT and OFFSET

LIMIT restricts how many rows a query returns. OFFSET skips a number of rows before starting to return results.

```sql
SELECT * FROM products ORDER BY price DESC LIMIT 5;
```

### LIMIT

Returns at most N rows. If the table has fewer rows than N, all rows are returned:

```sql
SELECT * FROM logs LIMIT 100;
```

Without ORDER BY, which rows you get is non-deterministic. Always pair LIMIT with ORDER BY when you need a specific "top N" result.

### OFFSET

Skips the first N rows before returning results:

```sql
SELECT * FROM products ORDER BY id LIMIT 10 OFFSET 20;
```

This skips the first 20 rows and returns rows 21 through 30. OFFSET 0 is the default and returns from the beginning.

### Pagination Pattern

LIMIT and OFFSET together implement pagination:

- Page 1: LIMIT 10 OFFSET 0
- Page 2: LIMIT 10 OFFSET 10
- Page N: LIMIT 10 OFFSET (N-1) * 10

### LIMIT Without ORDER BY

A common mistake is using LIMIT without ORDER BY. The database picks whichever rows are fastest to retrieve, which can change between executions. For reproducible results, always sort first.

### Execution Order

LIMIT and OFFSET are processed last, after ORDER BY. The full order: FROM then WHERE then SELECT then ORDER BY then LIMIT/OFFSET.
