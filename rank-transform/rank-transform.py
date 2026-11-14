def rank_transform(values: list) -> list:
    """
    Returns the one-based average rank of every value.
    """
    n = len(values)
    # Pair each value with its original index and sort by value ascending
    indexed_values = sorted((val, idx) for idx, val in enumerate(values))
    
    ranks = [0.0] * n
    i = 0
    
    while i < n:
        # Find the end of the current group of tied values
        j = i
        while j < n and indexed_values[j][0] == indexed_values[i][0]:
            j += 1
        
        # Positions occupied are 1-based from (i + 1) to j
        # The average of consecutive integers from (i + 1) to j is ((i + 1) + j) / 2.0
        avg_rank = (i + 1 + j) / 2.0
        
        # Assign this average rank to all original indices in the tied group
        for k in range(i, j):
            original_idx = indexed_values[k][1]
            ranks[original_idx] = avg_rank
            
        i = j
        
    return ranks
