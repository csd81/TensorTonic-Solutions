-- Returns: name, display_email, status.
SELECT 
    name,
    COALESCE(email, 'N/A') AS display_email,
    CASE 
        WHEN deactivated_at IS NULL THEN 'active'
        ELSE 'inactive'
    END AS status
FROM 
    customers
WHERE 
    phone IS NOT NULL
ORDER BY 
    name ASC;
