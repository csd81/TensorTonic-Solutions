-- Returns: username, session_count, activity_level, platform_type.
SELECT 
    username,
    session_count,
    CASE 
        WHEN session_count >= 50 THEN 'Power'
        WHEN session_count >= 10 AND session_count <= 49 THEN 'Casual'
        ELSE 'Dormant'
    END AS activity_level,
    CASE 
        WHEN platform IN ('ios', 'android') THEN 'Mobile'
        WHEN platform IN ('web', 'desktop') THEN 'Desktop'
        ELSE 'Other'
    END AS platform_type
FROM 
    user_sessions
ORDER BY 
    CASE activity_level
        WHEN 'Power' THEN 1
        WHEN 'Casual' THEN 2
        WHEN 'Dormant' THEN 3
        ELSE 4
    END,
    username ASC;
