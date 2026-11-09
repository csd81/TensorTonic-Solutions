## Multiple Joins

When data is split across more than two tables, you chain multiple JOIN clauses in a single query. Each JOIN adds one more table to the result, connecting it via a shared key column.

### Syntax for Chaining Joins

```sql
SELECT a.col1, b.col2, c.col3
FROM table_a a
JOIN table_b b ON a.id = b.a_id
JOIN table_c c ON a.id = c.a_id;
```

Each JOIN clause specifies which table to add and the ON condition that links it to a table already in the query. You can join the new table to any table that appeared earlier - not only the first one.

### Join Types in Multi-Table Queries

Every join in the chain has its own type (INNER, LEFT, RIGHT, FULL). You choose the type independently for each join based on whether you want to keep unmatched rows from that particular table:

```sql
SELECT u.name, o.order_id, p.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id
LEFT JOIN payments p ON o.id = p.order_id;
```

Here the first join is INNER (only users with orders), while the second is LEFT (keep orders even if they have no payment record yet).

### INNER JOIN Behavior

An INNER JOIN returns only rows where the ON condition matches in both tables. If a row in the left table has no matching row in the right table, that row is excluded entirely:

```sql
-- Only returns users who have at least one order
SELECT u.name, o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id;
```

When you chain two INNER JOINs, a row must have matches in all three tables to appear in the result. This is useful when you want to find records that exist across every table.

### LEFT JOIN vs INNER JOIN

A LEFT JOIN keeps all rows from the left table, filling in NULLs for columns from the right table when there is no match:

```sql
-- Returns ALL users, with NULL for total when they have no orders
SELECT u.name, o.total
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
```

Choosing between LEFT JOIN and INNER JOIN depends on the question you are answering:

- "Show me users who placed orders" - use INNER JOIN
- "Show me all users and their orders if any" - use LEFT JOIN

### Table Aliases

When joining multiple tables, aliases keep the query readable and resolve column name ambiguity:

```sql
SELECT u.name, ea.experiment_name, c.revenue
FROM users u
INNER JOIN experiment_assignments ea ON u.id = ea.user_id
INNER JOIN conversions c ON u.id = c.user_id;
```

### Join Order

In standard SQL the optimizer is free to reorder joins for performance. The order you write them in does not affect the result set (for INNER JOINs). However, the logical reading order matters for clarity: start with the primary entity and add related tables in a natural sequence.

For LEFT JOINs, order does matter semantically - the "left" table is the one that appears before the JOIN keyword.

### Common Pitfalls

**Missing a join**: If you forget to join one of the required tables, you either get a missing column error or a Cartesian product (every row from one table paired with every row from another).

**Using LEFT JOIN when INNER JOIN is needed**: This produces extra rows with NULLs for unmatched records. If your question asks for rows that exist in all tables, use INNER JOIN.

**Ambiguous column names**: When two tables share a column name (e.g., id), you must qualify it with the table alias: u.id vs ea.id. Omitting the alias causes an ambiguity error.

### Star Schema Pattern

In data analysis, a common layout is one central "fact" table surrounded by "dimension" tables. Queries join the fact table to multiple dimensions:

```sql
SELECT d.date_str, p.product_name, SUM(s.quantity)
FROM sales s
INNER JOIN dates d ON s.date_id = d.id
INNER JOIN products p ON s.product_id = p.id
GROUP BY d.date_str, p.product_name;
```

This pattern scales to any number of dimension tables - just add more JOIN clauses.