def quadratic_minimum(a: float, b: float, c: float) -> dict:
    """
    Returns the minimizer and minimum value in a dictionary.
    """
    # Calculate the minimizer (vertex x-coordinate)
    x_star = -b / (2 * a)
    
    # Calculate the minimum function value at x_star
    f_min = c - (b ** 2) / (4 * a)
    
    # Return rounded values as per requirements
    return {
        "x_star": round(float(x_star), 6),
        "f_min": round(float(f_min), 6)
    }
