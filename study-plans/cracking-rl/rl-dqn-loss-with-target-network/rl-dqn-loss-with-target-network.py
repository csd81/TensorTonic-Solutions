def dqn_loss(Q_online: list, Q_target_next: list, actions: list, rewards: list, dones: list, gamma: float) -> float:
    """
    Returns the mean squared TD loss, rounded to four decimals.
    """
    B = len(actions)
    total_squared_error = 0.0
    
    for i in range(B):
        r = rewards[i]
        d = dones[i]
        a = actions[i]
        
        # Get the Q-value prediction from the online network for the taken action
        q_online_val = Q_online[i][a]
        
        # Calculate max_a Q_target(s'_i, a) for the next state
        max_q_target = max(Q_target_next[i])
        
        # Compute the target value y_i
        y = r + gamma * (1 - d) * max_q_target
        
        # Accumulate the squared error
        total_squared_error += (y - q_online_val) ** 2
        
    return round(total_squared_error / B, 4)
