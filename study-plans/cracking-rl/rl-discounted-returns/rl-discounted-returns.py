def discounted_returns(rewards: list, gamma: float) -> list:
    """
    Returns one discounted return per reward, rounded to four decimals.
    """
    T = len(rewards)
    returns = [0.0] * T
    
    # Start from the last element and move backwards
    current_return = 0.0
    for t in reversed(range(T)):
        current_return = rewards[t] + gamma * current_return
        returns[t] = round(current_return, 4)
        
    return returns
