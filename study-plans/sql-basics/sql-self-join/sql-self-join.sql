-- Returns: username, referrer_name.
SELECT 
    u.username AS username,
    COALESCE(r.username, 'organic') AS referrer_name
FROM 
    user_referrals u
LEFT JOIN 
    user_referrals r ON u.referred_by = r.id
ORDER BY 
    CASE WHEN u.username IS NULL THEN 1 ELSE 0 END ASC,
    u.username ASC;
