-- Returns: name, salary.
SELECT name, salary
FROM employees
WHERE (department = 'Engineering' OR department = 'Marketing')
  AND salary > 70000;
