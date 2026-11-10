def per_priorities_and_weights(td_errors: list, alpha: float, beta: float, epsilon: float = 1e-6) -> tuple:
    """
    Returns (probs, is_weights), two float lists rounded to four decimals.
    """
    N = len(td_errors)
    
    # 1. Compute unnormalized priorities: p_i = (|delta_i| + epsilon) ^ alpha
    p = [(abs(delta) + epsilon) ** alpha for delta in td_errors]
    
    # 2. Compute proportional sampling probabilities: P_i = p_i / sum(p_j)
    sum_p = sum(p)
    probs = [p_i / sum_p for p_i in p]
    
    # 3. Compute raw importance-sampling weights: w_i = (N * P_i) ^ (-beta)
    w = [(N * P_i) ** (-beta) for P_i in probs]
    
    # 4. Normalize importance-sampling weights by max(w_j)
    max_w = max(w)
    is_weights = [w_i / max_w for w_i in w]
    
    # 5. Round final results to 4 decimal places
    probs_rounded = [round(prob, 4) for prob in probs]
    is_weights_rounded = [round(weight, 4) for weight in is_weights]
    
    return probs_rounded, is_weights_rounded
