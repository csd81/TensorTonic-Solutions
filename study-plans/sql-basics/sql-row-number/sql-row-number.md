## ROW_NUMBER Window Function

ROW_NUMBER() is a window function that assigns a unique sequential integer to each row within a partition. Unlike RANK() or DENSE_RANK(), it never produces duplicates - every row gets a distinct number, even when values are tied.

### Basic Syntax

```sql
ROW_NUMBER() OVER (
    [PARTITION BY column1, column2, ...]
    ORDER BY columnA [ASC|DESC], columnB [ASC|DESC], ...
) AS alias
```

- **PARTITION BY** (optional): Divides the result set into groups. Row numbering restarts at 1 for each partition.
- **ORDER BY**: Determines the numbering sequence within each partition. DuckDB permits an unordered window, but this task requires the stated score and username order.

### Simple Example

```sql
SELECT name, score,
       ROW_NUMBER() OVER (ORDER BY score DESC) AS rank
FROM players;
```

With data (Alice, 90), (Bob, 85), (Charlie, 90), this might produce:

| name    | score | rank |
|---------|-------|------|
| Alice   | 90    | 1    |
| Charlie | 90    | 2    |
| Bob     | 85    | 3    |

Notice that Alice and Charlie both have 90, but they get different row numbers (1 and 2). The exact order between tied rows is non-deterministic unless you add a tiebreaker column to the ORDER BY.

### PARTITION BY

Without PARTITION BY, the entire result set is treated as one partition. With it, numbering restarts independently for each group:

```sql
SELECT department, name, salary,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
```

Each department gets its own sequence: rank 1 is the highest-paid person in that department, rank 2 is the second-highest, and so on.

### ROW_NUMBER vs RANK vs DENSE_RANK

| score | ROW_NUMBER | RANK | DENSE_RANK |
|-------|-----------|------|------------|
| 100   | 1         | 1    | 1          |
| 95    | 2         | 2    | 2          |
| 95    | 3         | 2    | 2          |
| 80    | 4         | 4    | 3          |

- **ROW_NUMBER**: Always unique. Tied rows get arbitrary distinct numbers.
- **RANK**: Tied rows share the same number, then the next number skips (2, 2, 4).
- **DENSE_RANK**: Tied rows share the same number, but the next number does not skip (2, 2, 3).

### Deterministic Tiebreaking

When rows have the same value in the ORDER BY column, ROW_NUMBER assigns them in an arbitrary order. To make results reproducible, add a tiebreaker column:

```sql
ROW_NUMBER() OVER (
    PARTITION BY segment
    ORDER BY score DESC, username ASC
) AS activity_rank
```

### Common Use Cases

- **Top-N per group**: Find the top 3 products by revenue in each category using WHERE rn <= 3 in a subquery.
- **Deduplication**: When a table has duplicate rows, assign row numbers partitioned by the key columns and keep only rn = 1.
- **Pagination**: Assign row numbers to the full result set and filter by range for page-based queries.
- **Sequential numbering**: Generate sequence numbers for reporting or display purposes.