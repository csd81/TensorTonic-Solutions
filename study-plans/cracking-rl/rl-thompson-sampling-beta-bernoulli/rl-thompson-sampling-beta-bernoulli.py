def thompson_posterior(K: int, observations: list, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> tuple:
    """
    Returns (alpha_list, beta_list), each with K floats rounded to four decimals.
    """
    # Initialize successes and failures tracking for each of the K arms
    successes = [0.0] * K
    failures = [0.0] * K
    
    # Aggregate outcomes from the observations list
    for arm, reward in observations:
        if reward == 1:
            successes[arm] += 1
        elif reward == 0:
            failures[arm] += 1
            
    # Calculate updated posterior parameters and round to 4 decimal places
    alpha_list = [round(alpha_prior + successes[i], 4) for i in range(K)]
    beta_list = [round(beta_prior + failures[i], 4) for i in range(K)]
    
    return (alpha_list, beta_list)
