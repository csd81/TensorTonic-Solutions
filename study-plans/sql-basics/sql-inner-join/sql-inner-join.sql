-- Returns: name, salary, dept_name.
SELECT 
    e.name, 
    e.salary, 
    d.dept_name
FROM 
    employees e
INNER JOIN 
    departments d ON e.dept_id = d.id
ORDER BY 
    e.name ASC;
