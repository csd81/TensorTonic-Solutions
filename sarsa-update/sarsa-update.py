def sarsa_update(q_table: list, state: int, action: int, reward: float, next_state: int, next_action: int, alpha: float, gamma: float) -> list:
    """
    Returns a copied Q-table after one SARSA update.
    """
    # Create a deep copy of the 2D list with values converted to floats
    updated_q = [[round(float(val), 4) for val in row] for row in q_table]
    
    # Calculate the TD error: r + gamma * Q(s', a') - Q(s, a)
    td_error = reward + gamma * q_table[next_state][next_action] - q_table[state][action]
    
    # Update Q(s, a) and round to 4 decimal places
    new_q_value = q_table[state][action] + alpha * td_error
    updated_q[state][action] = round(float(new_q_value), 4)
    
    return updated_q
