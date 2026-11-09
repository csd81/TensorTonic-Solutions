-- Returns: model_name, dataset, accuracy, accuracy_rank, accuracy_dense_rank.
SELECT 
    model_name,
    dataset,
    accuracy,
    RANK() OVER (PARTITION BY dataset ORDER BY accuracy DESC) AS accuracy_rank,
    DENSE_RANK() OVER (PARTITION BY dataset ORDER BY accuracy DESC) AS accuracy_dense_rank
FROM 
    model_metrics
ORDER BY 
    dataset ASC, 
    accuracy DESC, 
    model_name ASC;
