import copy

def expected_sarsa_update(Q: list, transitions: list, policy: list, alpha: float, gamma: float) -> list:
    """
    Applies Expected SARSA updates using the supplied action probabilities.
    Returns a new nested list with the same shape as Q, rounded to 4 decimal places.
    """
    # Create a deep copy to leave the original Q-table unchanged
    Q_working = copy.deepcopy(Q)
    
    for s, a, r, s_next in transitions:
        # Calculate the expected Q-value for the next state under the current policy
        next_expectation = sum(
            policy[s_next][a_prime] * Q_working[s_next][a_prime] 
            for a_prime in range(len(Q_working[s_next]))
        )
        
        # Calculate the TD target and update the current state-action pair
        td_target = r + gamma * next_expectation
        Q_working[s][a] += alpha * (td_target - Q_working[s][a])
        
    # Cast explicitly to float() after rounding to ensure 1.0 and 0.0 formatting matches
    return [[float(round(val, 4)) for val in row] for row in Q_working]
