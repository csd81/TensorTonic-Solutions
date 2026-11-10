def thompson_posterior(K: int, observations: list, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> tuple:
    """
    Returns (alpha_list, beta_list), each with K floats rounded to four decimals.
    """
    # Initialize counts for successes (S) and failures (F) for each arm
    successes = [0] * K
    failures = [0] * K
    
    # Aggregate observations
    for arm, reward in observations:
        if reward == 1:
            successes[arm] += 1
        elif reward == 0:
            failures[arm] += 1
            
    # Calculate posterior parameters and round to 4 decimal places
    alpha_list = [round(float(alpha_prior + successes[i]), 4) for i in range(K)]
    beta_list = [round(float(beta_prior + failures[i]), 4) for i in range(K)]
    
    return (alpha_list, beta_list)
