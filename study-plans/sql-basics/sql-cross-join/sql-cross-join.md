## Cross Join

A CROSS JOIN produces the Cartesian product of two tables: every row from the left table is paired with every row from the right table. If the left table has N rows and the right table has M rows, the result contains N x M rows.

### Syntax

```sql
SELECT a.col1, b.col2
FROM table_a a
CROSS JOIN table_b b;
```

There is an older implicit syntax that achieves the same result by listing tables separated by commas in the FROM clause:

```sql
SELECT a.col1, b.col2
FROM table_a a, table_b b;
```

Both forms produce identical results, but the explicit CROSS JOIN keyword is preferred because it signals intent clearly - the reader immediately knows a Cartesian product is deliberate rather than a forgotten join condition.

### When to Use CROSS JOIN

Cross joins are useful when you need to generate every possible combination of values from two sets:

- **Report templates**: Pair every region with every time period to create a grid that can later be filled with actual data.
- **Feature matrices**: Combine user segments with metrics so that every cell in a dashboard has a placeholder row.
- **Calendar scaffolding**: Cross join a list of dates with a list of stores to produce a row for every store-day, ensuring no gaps in time-series analysis.
- **Test data generation**: Combine parameter sets to produce all permutations for exhaustive testing.

### Cross Join vs Other Joins

A cross join differs from INNER JOIN, LEFT JOIN, and other conditional joins in one fundamental way: it has **no ON clause**. Other joins match rows based on a condition (typically a key equality); a cross join matches every row with every other row unconditionally.

If you accidentally write an INNER JOIN with an equality condition between unrelated columns, you will get only rows where those columns happen to share a value - a tiny subset of the Cartesian product rather than the full grid.

### Performance Considerations

Because the output size is N x M, cross joins can produce very large result sets. Crossing a 1,000-row table with a 10,000-row table yields 10 million rows. Always verify that both input tables are small enough that the Cartesian product is manageable.

### Ordering the Result

A cross join does not guarantee any particular row order. To get deterministic output, always add an explicit ORDER BY clause. For grid-style reports, ordering by both columns alphabetically or by some priority field is standard practice.