-- Returns: respondent_clean, answer_length, answer_preview, source_domain.
SELECT 
    LOWER(TRIM(respondent)) AS respondent_clean,
    LENGTH(TRIM(raw_answer)) AS answer_length,
    SUBSTR(TRIM(raw_answer), 1, 20) AS answer_preview,
    SPLIT_PART(source_url, '/', 3) AS source_domain
FROM 
    survey_responses
ORDER BY 
    respondent_clean ASC;
