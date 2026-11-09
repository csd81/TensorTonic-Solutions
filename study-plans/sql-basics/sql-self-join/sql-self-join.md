## SQL Self Joins

A self join is a join where a table is joined to itself. This is useful when rows in the same table have a parent-child, predecessor-successor, or peer relationship encoded via a foreign key that references the table's own primary key.

### Basic Syntax

Because the same table appears twice, each occurrence needs a unique alias:

```sql
SELECT a.col1, b.col2
FROM my_table a
JOIN my_table b ON a.some_fk = b.id;
```

a and b are aliases for the same physical table. The query engine treats them as two independent copies for the purpose of the join.

### INNER vs LEFT Self Join

```sql
-- INNER self join: only returns rows where a match exists on both sides
SELECT u.username, r.username AS referrer_name
FROM user_referrals u
JOIN user_referrals r ON u.referred_by = r.id;
-- Users with referred_by = NULL are DROPPED
```

```sql
-- LEFT self join: keeps all rows from the left table
SELECT u.username, r.username AS referrer_name
FROM user_referrals u
LEFT JOIN user_referrals r ON u.referred_by = r.id;
-- Users with no referrer get NULL for r.username
```

When the foreign key column is nullable, a LEFT JOIN is essential to retain unmatched rows.

### Handling NULLs with COALESCE

After a LEFT JOIN, unmatched rows produce NULL values. COALESCE replaces NULL with a default:

```sql
SELECT u.username,
       COALESCE(r.username, 'organic') AS referrer_name
FROM user_referrals u
LEFT JOIN user_referrals r ON u.referred_by = r.id;
```

### Common Use Cases

**Referral chains and network analysis:**

```sql
-- Two-level referral chain
SELECT u.username,
       r1.username AS referrer,
       r2.username AS referrers_referrer
FROM user_referrals u
LEFT JOIN user_referrals r1 ON u.referred_by = r1.id
LEFT JOIN user_referrals r2 ON r1.referred_by = r2.id;
```

**Hierarchy traversal:** Organizational charts with a manager_id column referencing the same table:

```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

**Finding pairs:** Self joins can pair rows - for example, users who share the same email domain:

```sql
SELECT a.username, b.username
FROM users a
JOIN users b ON SPLIT_PART(a.email, '@', 2) = SPLIT_PART(b.email, '@', 2)
  AND a.id < b.id;  -- avoid duplicate pairs and self-pairs
```

### Aliasing Rules

When self-joining, you **must** alias at least one occurrence of the table. Best practice is to alias both with names that convey each copy's role. Without aliases, the database cannot disambiguate column references and the query will fail.

### Performance Considerations

Self joins are not inherently expensive. However, on large tables ensure the join column and foreign key column are indexed. For deep hierarchies (more than 2-3 levels), consider recursive CTEs instead of chaining multiple self joins.