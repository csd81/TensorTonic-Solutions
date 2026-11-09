## CASE Statements

The CASE expression is SQL's way of implementing conditional (if/then/else) logic inline within a query. It can appear in SELECT, WHERE, ORDER BY, GROUP BY, and even inside aggregate functions.

### Two Forms of CASE

#### Searched CASE (most common)

Evaluates a series of boolean conditions in order and returns the value for the first condition that is true:

```sql
CASE
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    ELSE default_result
END
```

Each WHEN clause can test completely different columns or expressions. The conditions are evaluated top-to-bottom; the first match wins. If no condition matches and there is no ELSE, the result is NULL.

#### Simple CASE

Compares a single expression against a set of values:

```sql
CASE expression
    WHEN value1 THEN result1
    WHEN value2 THEN result2
    ELSE default_result
END
```

This is syntactic sugar for a searched CASE where every condition is expression = valueN. It is more concise when you are comparing one column against discrete values, but it cannot handle range comparisons like >= 50.

### CASE in SELECT

The most common use - derive a new computed column:

```sql
SELECT username,
       CASE WHEN session_count >= 50 THEN 'Power User' ELSE 'Regular' END AS user_type
FROM user_sessions;
```

Always alias the CASE expression with AS so the output column has a meaningful name.

### CASE in WHERE

Filter rows based on conditional logic:

```sql
SELECT *
FROM orders
WHERE CASE WHEN priority = 'rush' THEN shipped_date <= due_date - 2
           ELSE shipped_date <= due_date
      END;
```

This is less common because the same logic can usually be expressed with AND/OR, but CASE can make complex conditional filters more readable.

### CASE in ORDER BY

Define a custom sort order that does not correspond to alphabetical or numeric ordering:

```sql
SELECT username, activity_level
FROM user_sessions
ORDER BY CASE activity_level
             WHEN 'Power'   THEN 1
             WHEN 'Casual'  THEN 2
             WHEN 'Dormant' THEN 3
             ELSE 4
         END;
```

By mapping each value to a numeric rank, you control the exact ordering.

### Evaluation Order Matters

CASE evaluates conditions top-to-bottom and short-circuits on the first match. When conditions overlap, put the most restrictive first:

```sql
-- Correct: checks >= 50 before >= 10
CASE WHEN session_count >= 50 THEN 'Power'
     WHEN session_count >= 10 THEN 'Casual'
     ELSE 'Dormant'
END

-- Wrong: everyone >= 10 matches first, including those >= 50
CASE WHEN session_count >= 10 THEN 'Casual'
     WHEN session_count >= 50 THEN 'Power'
     ELSE 'Dormant'
END
```

### ELSE Clause

The ELSE is optional but recommended. Without it, any row that does not match any WHEN condition gets NULL, which can cause unexpected behavior in downstream logic.

### CASE Inside Aggregates

CASE is often combined with aggregate functions for conditional aggregation:

```sql
SELECT platform_type,
       COUNT(CASE WHEN session_count >= 50 THEN 1 END) AS power_users,
       COUNT(CASE WHEN session_count < 10 THEN 1 END) AS dormant_users
FROM user_sessions
GROUP BY platform_type;
```

This pattern replaces multiple filtered subqueries with a single pass over the data.