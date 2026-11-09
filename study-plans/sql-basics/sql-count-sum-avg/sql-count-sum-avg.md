## Aggregate Functions: COUNT, SUM, AVG

COUNT, SUM, and AVG are the three most commonly used aggregate functions in SQL. Each has specific behavior with NULL values that you must understand.

### COUNT(*) vs COUNT(col)

- COUNT(*) counts every row in the group, including rows with NULLs.
- COUNT(col) counts only rows where col is not NULL.

```sql
-- Table: [10, NULL, 30]
SELECT COUNT(*), COUNT(value) FROM t;
-- Result: 3, 2
```

### SUM

SUM adds all non-NULL values. If all values are NULL, SUM returns NULL (not 0).

```sql
-- Table: [10, NULL, 30]
SELECT SUM(value) FROM t;
-- Result: 40
```

### AVG

AVG computes the mean of non-NULL values. It equals SUM(col) / COUNT(col), not SUM(col) / COUNT(*). This means NULL rows are excluded from both the numerator and denominator.

```sql
-- Table: [10, NULL, 30]
SELECT AVG(value) FROM t;
-- Result: 20.0 (not 13.33)
```

### ROUND for Clean Output

Aggregate results often produce long decimals. Use ROUND(value, n) to limit decimal places:

```sql
SELECT ROUND(AVG(price), 2) AS avg_price FROM products;
```

### NULL Gotchas

- SUM of an empty group or all-NULL column returns NULL, not 0
- Use COALESCE(SUM(col), 0) if you need 0 instead of NULL
- AVG excludes NULLs from the count, which can produce surprising results if you expect them to count as zero