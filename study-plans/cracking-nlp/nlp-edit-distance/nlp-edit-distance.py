def edit_distance(source: str, target: str) -> int:
    """
    Returns the minimum edit count as an integer.
    """
    m, n = len(source), len(target)
    
    # Create a 1D array to store the current row's distances.
    # This optimizes space from O(m*n) to O(n).
    dp = list(range(n + 1))
    
    for i in range(1, m + 1):
        # prev tracks the top-left diagonal value: D(i-1, j-1)
        prev = dp[0]
        dp[0] = i
        
        for j in range(1, n + 1):
            temp = dp[j]
            
            # Indicator is 0 if characters match, 1 otherwise
            # Remember 0-based indexing for string characters
            cost = 0 if source[i - 1] == target[j - 1] else 1
            
            # Compute the minimum of:
            # 1. dp[j] + 1     (Deletion)
            # 2. dp[j-1] + 1   (Insertion)
            # 3. prev + cost   (Substitution/Match)
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            
            prev = temp
            
    return dp[n]
