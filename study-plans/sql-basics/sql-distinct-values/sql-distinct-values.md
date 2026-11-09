## DISTINCT

DISTINCT removes duplicate rows from the result set. Two rows are considered duplicates when every selected column has the same value.

```sql
SELECT DISTINCT department FROM employees;
```

### How DISTINCT Works

Without DISTINCT, a query returns every row that matches. With DISTINCT, the database groups identical rows and returns only one representative from each group:

```sql
-- Without DISTINCT: returns 1000 rows, many repeated departments
SELECT department FROM employees;

-- With DISTINCT: returns only unique department names
SELECT DISTINCT department FROM employees;
```

### DISTINCT on Multiple Columns

When you select multiple columns, DISTINCT considers the combination of all selected columns. Two rows are duplicates only if every column matches:

```sql
SELECT DISTINCT department, job_title FROM employees;
```

This returns unique (department, job_title) pairs. The same department can appear multiple times if paired with different job titles.

### COUNT(DISTINCT column)

You can count unique values without listing them:

```sql
SELECT COUNT(DISTINCT department) AS num_departments FROM employees;
```

This returns a single number - how many unique non-null departments exist. COUNT(DISTINCT column) excludes NULL values.

### DISTINCT and NULLs

DISTINCT treats all NULL values as equal to each other. If a column has multiple NULL entries, DISTINCT keeps only one NULL in the output.

### Performance Note

DISTINCT requires the database to compare and deduplicate rows, which can be expensive on large tables. If you know your query already produces unique rows (e.g., selecting a primary key), DISTINCT is unnecessary overhead.
