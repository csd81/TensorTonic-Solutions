def reinforce_loss(log_probs: list, returns: list) -> float:
    """
    Returns the REINFORCE loss as a float rounded to four decimals.
    """
    T = len(log_probs)
    total_loss = 0.0
    
    for t in range(T):
        total_loss += log_probs[t] * returns[t]
        
    return round(-total_loss / T, 4)
