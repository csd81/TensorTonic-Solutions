def bayes_theorem(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
    """
    Returns the posterior probability rounded to four decimals.
    """
    # Step 1: Compute the complement prior P(A^c)
    p_not_a = 1.0 - p_a
    
    # Step 2: Compute the total probability of evidence P(B)
    p_b = (p_b_given_a * p_a) + (p_b_given_not_a * p_not_a)
    
    # Step 3: Divide the joint probability P(A and B) by P(B)
    posterior = (p_b_given_a * p_a) / p_b
    
    # Step 4: Return rounded to four decimal places
    return round(posterior, 4)
