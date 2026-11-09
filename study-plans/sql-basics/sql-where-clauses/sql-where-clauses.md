## The WHERE Clause

WHERE filters rows before they reach the result set. Only rows that satisfy the condition are returned.

```sql
SELECT * FROM employees WHERE department = 'Engineering';
```

### Comparison Operators

Standard operators work on numbers, strings, and dates:

| Operator | Meaning |
|----------|---------|
| = | Equal |
| <> or != | Not equal |
| <, > | Less / greater than |
| <=, >= | Less / greater than or equal |

### Combining Conditions

- AND - both conditions must be true
- OR - at least one condition must be true
- NOT - negates a condition

AND binds tighter than OR. Use parentheses to control evaluation order:

```sql
WHERE (status = 'active' OR status = 'pending') AND age > 18
```

### IN, BETWEEN, LIKE

- IN (v1, v2, v3) - matches any value in the list. Cleaner than chaining multiple ORs.
- BETWEEN low AND high - inclusive range check. Equivalent to col >= low AND col <= high.
- LIKE - pattern matching. % matches zero or more characters, _ matches exactly one.

```sql
WHERE name LIKE 'A%'      -- starts with A
WHERE name LIKE '%son'     -- ends with son
WHERE name LIKE '_a%'      -- second character is a
```

### NULL Checks

NULL represents missing data. You cannot use = or <> with NULL - they always return unknown. Use IS NULL or IS NOT NULL instead:

```sql
WHERE email IS NOT NULL
```

### Execution Order

WHERE filters rows before the final result is selected, so it can use columns absent from the output. DuckDB also permits non-aggregate SELECT aliases in WHERE.
