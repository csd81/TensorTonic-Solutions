## LAG and LEAD Window Functions

LAG and LEAD let you access values from other rows relative to the current row, without needing a self-join. They are essential for comparing consecutive rows - calculating differences, growth rates, or detecting changes.

### LAG()

LAG looks backward. It returns a value from a previous row in the window:

```sql
LAG(column, offset, default) OVER (ORDER BY ...)
```

- column: the value to retrieve
- offset: how many rows back (default is 1)
- default: value to return when there is no previous row (default is NULL)

```sql
LAG(revenue, 1, 0) OVER (ORDER BY month)
-- For each row: get the revenue from 1 row before
-- First row has no predecessor, so it returns 0
```

### LEAD()

LEAD looks forward. It returns a value from a subsequent row:

```sql
LEAD(column, offset, default) OVER (ORDER BY ...)
```

```sql
LEAD(revenue, 1, 0) OVER (ORDER BY month)
-- For each row: get the revenue from 1 row ahead
-- Last row has no successor, so it returns 0
```

### The Default Parameter

Without a default, LAG and LEAD return NULL for edge rows (first row for LAG, last row for LEAD). Providing a default avoids NULL arithmetic problems:

```sql
-- Without default: first row gets NULL, so subtraction yields NULL
revenue - LAG(revenue) OVER (ORDER BY month)

-- With default 0: first row gets 0, subtraction works cleanly
revenue - LAG(revenue, 1, 0) OVER (ORDER BY month)
```

### PARTITION BY with LAG/LEAD

Add PARTITION BY to compare rows within groups independently:

```sql
LAG(revenue) OVER (PARTITION BY region ORDER BY month)
-- Compares each month to the previous month within the same region
```

### Common Use Cases

- **Period-over-period change**: revenue - LAG(revenue, 1, 0) OVER (ORDER BY month)
- **Growth rate**: When previous revenue is nonzero, express revenue change as a percentage of that previous revenue.
- **Next value preview**: LEAD(status) OVER (ORDER BY event_time) to see what happens next
- **Gap detection**: compare LEAD(date) with current date to find missing periods
