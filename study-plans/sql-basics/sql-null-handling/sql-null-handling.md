## NULL Handling in SQL

NULL represents a missing or unknown value. It is not the same as zero, an empty string, or false. NULL requires special handling because it behaves differently from regular values in comparisons, functions, and logic.

### NULL in Comparisons

Any comparison with NULL returns NULL (not true or false):

```sql
-- These are ALL unknown (not true, not false):
NULL = NULL
NULL != 1
NULL > 0
```

To check for NULL, use IS NULL or IS NOT NULL:

```sql
WHERE email IS NULL       -- rows with missing email
WHERE email IS NOT NULL   -- rows with an email
```

### COALESCE

COALESCE returns the first non-NULL argument. It is the standard way to provide a fallback value:

```sql
COALESCE(email, 'N/A')         -- email if present, otherwise 'N/A'
COALESCE(nickname, first_name) -- nickname if present, otherwise first_name
```

You can chain multiple arguments. COALESCE stops at the first non-NULL value.

### NULLIF

NULLIF(a, b) returns NULL if a = b, otherwise returns a. Useful for turning sentinel values into proper NULLs:

```sql
NULLIF(discount, 0)   -- treats 0 discount as NULL
NULLIF(status, '')     -- treats empty string as NULL
```

### CASE with NULL

Use IS NULL in CASE expressions, not = NULL:

```sql
CASE
  WHEN deactivated_at IS NULL THEN 'active'
  ELSE 'inactive'
END
```

### NULL in Aggregates

Most aggregate functions ignore NULL values:

- COUNT(column) skips NULLs; COUNT(*) counts all rows
- SUM, AVG, MIN, MAX all skip NULLs
- If all values are NULL, the aggregate returns NULL (except COUNT which returns 0)
