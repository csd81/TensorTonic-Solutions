## RANK and DENSE_RANK

RANK() and DENSE_RANK() are window functions that assign a ranking to each row within a partition of a result set. Both are commonly used to compare items within groups - for example, ranking ML models by accuracy within each evaluation dataset.

### RANK()

RANK() assigns a rank to each row within a partition, with tied rows receiving the same rank. After a tie, the next rank **skips** by the number of tied rows:

```sql
RANK() OVER (PARTITION BY group_col ORDER BY value_col DESC) AS rnk
```

If two rows are tied at rank 1, the next row gets rank 3 (not 2). This means the rank values correspond to the row's position in the sorted list.

Example output with a tie at rank 2:

| model   | accuracy | rank |
|---------|----------|------|
| ModelA  | 95       | 1    |
| ModelB  | 88       | 2    |
| ModelC  | 88       | 2    |
| ModelD  | 75       | 4    |

### DENSE_RANK()

DENSE_RANK() also assigns the same rank to tied rows, but the next rank after a tie is always the next consecutive integer - no gaps:

```sql
DENSE_RANK() OVER (PARTITION BY group_col ORDER BY value_col DESC) AS dense_rnk
```

Same data as above:

| model   | accuracy | dense_rank |
|---------|----------|------------|
| ModelA  | 95       | 1          |
| ModelB  | 88       | 2          |
| ModelC  | 88       | 2          |
| ModelD  | 75       | 3          |

### ROW_NUMBER() - How It Differs

ROW_NUMBER() assigns a unique sequential integer to each row. Tied rows receive different numbers (the assignment among ties is non-deterministic unless you add a tiebreaker to ORDER BY):

| model   | accuracy | row_number |
|---------|----------|------------|
| ModelA  | 95       | 1          |
| ModelB  | 88       | 2          |
| ModelC  | 88       | 3          |
| ModelD  | 75       | 4          |

### PARTITION BY

The PARTITION BY clause divides the result set into groups. Ranking restarts within each partition:

```sql
RANK() OVER (PARTITION BY dataset ORDER BY accuracy DESC)
```

This ranks models independently within each dataset. Without PARTITION BY, all rows are treated as a single partition and ranked together.

### ORDER BY Within the Window

The ORDER BY inside OVER(...) determines the ranking order - it is independent of the query-level ORDER BY:

- ORDER BY accuracy DESC - highest accuracy gets rank 1
- ORDER BY accuracy ASC - lowest accuracy gets rank 1

### When to Use Which

- **RANK**: When you need the rank to reflect the actual position in the sorted list. Useful for competition-style ranking ("1st place, 1st place, 3rd place").
- **DENSE_RANK**: When you want consecutive rank values without gaps. Useful for bucketing or when the count of distinct rank levels matters ("top 3 accuracy tiers").
- **ROW_NUMBER**: When you need a unique identifier per row regardless of ties. Useful for pagination or selecting exactly N rows.

### Combining Multiple Window Functions

You can include multiple ranking functions in a single SELECT:

```sql
SELECT model_name, accuracy,
       RANK() OVER (ORDER BY accuracy DESC) AS rnk,
       DENSE_RANK() OVER (ORDER BY accuracy DESC) AS dense_rnk,
       ROW_NUMBER() OVER (ORDER BY accuracy DESC) AS row_num
FROM models;
```

Each function computes independently but can share the same window specification.