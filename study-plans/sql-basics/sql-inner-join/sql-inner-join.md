## INNER JOIN

An INNER JOIN combines rows from two tables where a condition (usually a key match) is satisfied. Rows that have no match in the other table are excluded entirely.

```sql
SELECT orders.id, customers.name
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id;
```

### How It Works

The database compares every row in the left table against every row in the right table. Only pairs where the ON condition evaluates to true appear in the output. If a left row has no matching right row, it is dropped. If a right row has no matching left row, it is also dropped.

### Table Aliases

When joining tables, queries get verbose. Aliases shorten them:

```sql
SELECT o.id, c.name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id;
```

The alias is declared right after the table name in FROM or JOIN. Once defined, use it everywhere in the query.

### Ambiguous Columns

When both tables have a column with the same name (e.g., id), you must prefix with the table name or alias. Without it, the database raises an ambiguous column error:

```sql
-- Ambiguous: both tables have 'id'
SELECT id FROM orders INNER JOIN customers ON ...

-- Fixed:
SELECT orders.id FROM orders INNER JOIN customers ON ...
```

### JOIN vs WHERE

These two queries are equivalent:

```sql
-- Explicit JOIN syntax (preferred)
SELECT * FROM a INNER JOIN b ON a.id = b.a_id;

-- Implicit join via WHERE (older style)
SELECT * FROM a, b WHERE a.id = b.a_id;
```

The explicit JOIN syntax is clearer about intent and harder to accidentally turn into a cross join.

### Multiple Matches

If a row in the left table matches multiple rows in the right table, it appears multiple times in the output - once per match. This is called row multiplication and is normal behavior.
