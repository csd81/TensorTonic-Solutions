def gae_advantages(rewards: list, values: list, gamma: float, lam: float, last_value: float = 0.0) -> list:
    """
    Returns the advantage list in timestep order, rounded to four decimals.
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    
    # Append last_value to state-values to act as V[T]
    V = values + [last_value]
    
    # Process the sequence backwards from T-1 down to 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * V[t+1] - V[t]
        gae = delta + gamma * lam * gae
        advantages[t] = round(gae, 4)
        
    return advantages
