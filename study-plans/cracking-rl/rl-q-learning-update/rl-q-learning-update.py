def q_learning_update(Q: list, transitions: list, alpha: float, gamma: float) -> list:
    """
    Returns a new updated Q-table as a nested list, rounded to four decimals.
    """
    # Create a deep copy of the Q-table to avoid mutating the original input
    updated_Q = [row[:] for row in Q]
    
    # Process each transition sequentially using the latest available Q-values
    for state, action, reward, next_state in transitions:
        # Find the maximum Q-value achievable from the next state across all actions
        max_next_q = max(updated_Q[next_state])
        
        # Calculate the Q-learning target: r + gamma * max_a' Q(s', a')
        target = reward + gamma * max_next_q
        
        # Apply the Bellman equation update rule
        updated_Q[state][action] += alpha * (target - updated_Q[state][action])
        
    # Round all values to 4 decimal places before returning
    return [[round(float(val), 4) for val in row] for row in updated_Q]
