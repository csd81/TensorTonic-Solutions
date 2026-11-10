def a2c_loss(
    log_probs: list, 
    advantages: list, 
    values: list, 
    returns: list, 
    entropies: list, 
    value_coef: float = 0.5, 
    entropy_coef: float = 0.01
) -> float:
    """
    Returns the total A2C loss as a float rounded to four decimals.
    """
    T = len(log_probs)
    
    # 1. Policy Loss: L_pi = -1/T * sum(ell_t * A_t)
    policy_loss = - (1 / T) * sum(lp * adv for lp, adv in zip(log_probs, advantages))
    
    # 2. Value Loss: L_V = 1/T * sum((G_t - V_t)^2)
    value_loss = (1 / T) * sum((ret - val) ** 2 for ret, val in zip(returns, values))
    
    # 3. Mean Entropy: H_bar = 1/T * sum(H_t)
    mean_entropy = (1 / T) * sum(entropies)
    
    # 4. Total Loss: L = L_pi + c_v * L_V - c_e * H_bar
    total_loss = policy_loss + (value_coef * value_loss) - (entropy_coef * mean_entropy)
    
    return round(total_loss, 4)
