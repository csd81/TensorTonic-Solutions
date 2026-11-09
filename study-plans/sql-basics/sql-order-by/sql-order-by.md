## The ORDER BY Clause

By default, SQL makes no guarantees about the order of rows in a result set. ORDER BY sorts the output deterministically.

```sql
SELECT name, salary FROM employees ORDER BY salary DESC;
```

### Sort Direction

- ASC (ascending) - smallest to largest. This is the default if you omit the direction.
- DESC (descending) - largest to smallest.

### Sorting on Multiple Columns

You can sort by several columns. The database sorts by the first column, then breaks ties using the second, and so on:

```sql
SELECT * FROM products ORDER BY category ASC, price DESC;
```

This groups products by category alphabetically, and within each category, lists the most expensive items first.

### NULLs in Sorting

DuckDB places NULL values last by default for both ascending and descending order. Use NULLS FIRST or NULLS LAST to choose their position explicitly:

```sql
ORDER BY column ASC NULLS FIRST
```

### Sorting by Expressions and Aliases

You can sort by computed expressions or by column aliases defined in SELECT:

```sql
SELECT name, price * quantity AS total FROM orders ORDER BY total DESC;
```

### Execution Order

ORDER BY is processed after SELECT, which is why it can reference column aliases. The full order so far: FROM then WHERE then SELECT then ORDER BY.
