def td_zero_update(V: list, transitions: list, alpha: float, gamma: float) -> list:
    """
    Returns the updated state-value list, rounded to four decimals.
    """
    # Create a copy of V using floats to prevent mutation of the input list
    V_updated = [float(val) for val in V]
    
    # Process each transition sequentially
    for s, r, s_prime in transitions:
        # Calculate the TD error: r + gamma * V(s') - V(s)
        td_error = r + gamma * V_updated[s_prime] - V_updated[s]
        
        # Update V(s) in place using the latest values
        V_updated[s] += alpha * td_error
        
    # Return the updated values rounded to four decimal places
    return [round(val, 4) for val in V_updated]
