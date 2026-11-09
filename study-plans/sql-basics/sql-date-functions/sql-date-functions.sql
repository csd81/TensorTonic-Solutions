-- Returns: username, signup_date, signup_year, signup_month, signup_quarter, cohort_month.
SELECT 
    username, 
    signup_date, 
    EXTRACT(YEAR FROM signup_date) AS signup_year, 
    EXTRACT(MONTH FROM signup_date) AS signup_month, 
    EXTRACT(QUARTER FROM signup_date) AS signup_quarter, 
    DATE_TRUNC('month', signup_date) AS cohort_month 
FROM signups 
ORDER BY signup_date ASC, username ASC;
