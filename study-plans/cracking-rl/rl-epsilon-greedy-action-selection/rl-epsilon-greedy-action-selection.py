def epsilon_greedy_probs(Q_values: list, epsilon: float) -> list:
    """
    Returns the action-probability list, rounded to four decimals.
    """
    # 1. Determine total number of actions
    A = len(Q_values)
    
    # 2. Identify the greedy action (returns the lowest index in case of a tie)
    greedy_idx = Q_values.index(max(Q_values))
    
    # 3. Calculate probabilities for each action index
    probs = []
    for a in range(A):
        if a == greedy_idx:
            prob = 1 - epsilon + (epsilon / A)
        else:
            prob = epsilon / A
        
        # 4. Round to four decimal places and append
        probs.append(round(prob, 4))
        
    return probs
