def independence_test(p_a: float, p_b: float, p_a_and_b: float) -> dict:
    """
    Returns the rounded product and independence decision in a dictionary.
    """
    # Compute the exact unrounded product
    unrounded_product = p_a * p_b
    
    # Check for independence using the specified strict tolerance
    is_independent = abs(p_a_and_b - unrounded_product) <= 0.000000001
    
    # Round only the product stored in the final result dictionary to 4 decimals
    return {
        "p_a_times_p_b": round(unrounded_product,4),
        "is_independent": is_independent
    }