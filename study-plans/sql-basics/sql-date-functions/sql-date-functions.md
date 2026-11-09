## Date Functions for Cohort Analysis

Cohort analysis groups users by the time period in which they signed up (or first performed some action), then tracks their behavior over subsequent periods. SQL date functions are essential for building cohorts because they let you normalize raw dates into consistent time buckets.

### EXTRACT

EXTRACT pulls a single numeric component out of a date or timestamp:

```sql
SELECT EXTRACT(YEAR FROM signup_date) AS yr,
       EXTRACT(MONTH FROM signup_date) AS mo,
       EXTRACT(QUARTER FROM signup_date) AS qtr
FROM signups;
```

Common fields: YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, QUARTER, DOW (day of week), DOY (day of year), WEEK, EPOCH.

The return type is always numeric. EXTRACT(MONTH FROM DATE '2024-03-15') returns 3, not 'March'.

### DATE_TRUNC - the Cohort Builder

DATE_TRUNC truncates a date or timestamp down to a specified precision, rounding it to the start of that period:

```sql
SELECT DATE_TRUNC('month', signup_date) AS cohort_month,
       DATE_TRUNC('quarter', signup_date) AS cohort_quarter,
       DATE_TRUNC('year', signup_date) AS cohort_year
FROM signups;
```

Given a date of 2024-07-18:
- DATE_TRUNC('month', ...) returns 2024-07-01
- DATE_TRUNC('quarter', ...) returns 2024-07-01 (Q3 starts July 1)
- DATE_TRUNC('year', ...) returns 2024-01-01

This is the workhorse of cohort analysis. By truncating every user's signup date to the first of their signup month, you create a cohort identifier that groups all users who signed up in the same month. You can then JOIN or GROUP BY this cohort identifier to measure retention, revenue, or other metrics over time.

A typical cohort retention query looks like:

```sql
SELECT DATE_TRUNC('month', s.signup_date) AS cohort_month,
       DATEDIFF('month', s.signup_date, a.activity_date) AS months_since_signup,
       COUNT(DISTINCT s.user_id) AS active_users
FROM signups s
JOIN activity a ON s.user_id = a.user_id
GROUP BY cohort_month, months_since_signup
ORDER BY cohort_month, months_since_signup;
```

Supported precision values include: 'microseconds', 'milliseconds', 'second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year', 'decade', 'century'.

### Quarter Numbering

The quarter number divides the year into four parts:

| Months    | Quarter |
|-----------|---------|
| Jan - Mar | 1       |
| Apr - Jun | 2       |
| Jul - Sep | 3       |
| Oct - Dec | 4       |

EXTRACT(QUARTER FROM DATE '2024-11-05') returns 4.

### DATE_PART

DATE_PART is functionally identical to EXTRACT in PostgreSQL and DuckDB but uses function-call syntax:

```sql
SELECT DATE_PART('year', signup_date) AS yr,
       DATE_PART('month', signup_date) AS mo
FROM signups;
```

In DuckDB the two forms are interchangeable. MySQL and SQL Server do not support DATE_PART - they use YEAR(), MONTH(), and similar named functions instead.

### Cross-Database Syntax Summary

| Operation          | PostgreSQL / DuckDB           | MySQL                      | SQL Server                   |
|--------------------|-------------------------------|----------------------------|------------------------------|
| Extract year       | EXTRACT(YEAR FROM d)          | YEAR(d)                    | YEAR(d) or DATEPART(year, d) |
| Extract month      | EXTRACT(MONTH FROM d)         | MONTH(d)                   | MONTH(d)                     |
| Truncate to month  | DATE_TRUNC('month', d)        | DATE_FORMAT(d, '%Y-%m-01') | DATEFROMPARTS(YEAR(d),MONTH(d),1) |
| Extract quarter    | EXTRACT(QUARTER FROM d)       | QUARTER(d)                 | DATEPART(quarter, d)         |

When writing portable SQL, be aware of these differences. This problem uses PostgreSQL / DuckDB syntax.
