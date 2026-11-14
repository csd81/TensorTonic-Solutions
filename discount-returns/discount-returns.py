def discount_returns(rewards: list, gamma: float) -> list:
    """
    Returns the discounted return at every timestep.
    """
    n = len(rewards)
    returns = [0.0] * n
    running_return = 0.0
    
    for t in range(n - 1, -1, -1):
        running_return = rewards[t] + gamma * running_return
        returns[t] = round(float(running_return), 4)
        
    return returns
