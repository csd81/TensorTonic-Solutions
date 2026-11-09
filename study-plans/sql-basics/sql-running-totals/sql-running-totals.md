## Running Totals with Window Functions

A **running total** (or cumulative sum) adds each row's value to the sum of all preceding rows in a defined order. SQL window functions make this straightforward.

### SUM() as a Window Function

When you use SUM(amount) OVER (...), SQL computes a sum for each row based on a **window frame** rather than collapsing rows into groups like GROUP BY does.

```sql
SUM(amount) OVER (ORDER BY txn_date)
```

With the default RANGE frame, each row includes all earlier dates and every row sharing its date. An explicit ROWS frame instead accumulates one ordered row at a time.

### PARTITION BY Resets the Running Total

Adding PARTITION BY account restarts the cumulative sum for each account:

```sql
SUM(amount) OVER (PARTITION BY account ORDER BY txn_date)
```

Row 1 of each account starts fresh. The running total never carries over from one account to the next.

### The Window Frame

When ORDER BY appears inside OVER(), the default frame is:

```
RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```

You can write this explicitly with ROWS instead of RANGE:

```
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```

ROWS counts individual rows, while RANGE groups rows with the same ORDER BY value together. When there are no ties, they behave identically. Using ROWS is more precise when you want a strict row-by-row accumulation.

### Why ORDER BY Matters

Without ORDER BY in the window, every row sees the same total (the full partition sum). The ordering is what turns a simple sum into a running total - it defines which rows are "before" the current row.
