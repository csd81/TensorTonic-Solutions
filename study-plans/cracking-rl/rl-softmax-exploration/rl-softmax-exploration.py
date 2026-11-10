import math

def softmax_action_probs(Q_values: list, tau: float) -> list:
    """
    Returns the action-probability list, rounded to four decimals.
    """
    # Step 1: Scale Q-values by the temperature tau
    scaled_Q = [q / tau for q in Q_values]
    
    # Step 2: Find the maximum scaled value for numerical stability
    max_scaled = max(scaled_Q)
    
    # Step 3: Compute stable numerators (exponential of shifted values)
    exps = [math.exp(sq - max_scaled) for sq in scaled_Q]
    
    # Step 4: Compute the denominator (sum of all numerators)
    sum_exps = sum(exps)
    
    # Step 5: Calculate probabilities and round to 4 decimal places
    probs = [round(e / sum_exps, 4) for e in exps]
    
    return probs
