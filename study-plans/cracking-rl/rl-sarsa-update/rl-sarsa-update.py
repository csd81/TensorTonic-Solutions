def sarsa_update(Q: list, transitions: list, alpha: float, gamma: float) -> list:
    """
    Returns the updated Q-table as a nested list, rounded to four decimals.
    """
    # Create a deep copy of the Q-table and cast elements to float to avoid type mismatches
    updated_Q = [[float(val) for val in row] for row in Q]
    
    # Process each transition sequentially
    for state, action, reward, next_state, next_action in transitions:
        # Fetch current Q-value and next state-action Q-value
        current_q = updated_Q[state][action]
        next_q = updated_Q[next_state][next_action]
        
        # Apply the SARSA TD update formula
        updated_Q[state][action] = current_q + alpha * (reward + gamma * next_q - current_q)
        
    # Round all values to 4 decimal places after all updates are complete
    rounded_Q = [[round(val, 4) for val in row] for row in updated_Q]
    return rounded_Q
