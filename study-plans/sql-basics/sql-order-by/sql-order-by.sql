-- Returns: name, subject, score.
SELECT name, subject, score
FROM students
ORDER BY 
    score DESC NULLS LAST, 
    name ASC NULLS LAST;
