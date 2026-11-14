def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """
    n = len(priorities)
    
    # 1. Compute powered priorities: p_i^alpha
    powered_priorities = [p ** alpha for p in priorities]
    
    # 2. Compute sampling probabilities: P(i) = p_i^alpha / sum(p_j^alpha)
    total_priority = sum(powered_priorities)
    probabilities = [p / total_priority for p in powered_priorities]
    
    # 3. Compute raw importance sampling weights: w_i = (N * P(i))^(-beta)
    raw_weights = [(n * prob) ** (-beta) for prob in probabilities]
    
    # 4. Normalize weights by the maximum weight: w_hat_i = w_i / max(w)
    max_weight = max(raw_weights)
    normalized_weights = [w / max_weight for w in raw_weights]
    
    return [probabilities, normalized_weights]
