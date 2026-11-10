def paraboloid_minimum(a: float, b: float, c: float, d: float, e: float) -> dict:
    """
    Returns the minimizer coordinates and minimum value in a dictionary.
    """
    x_star = -c / (2.0 * a)
    y_star = -d / (2.0 * b)
    f_min = a * (x_star ** 2) + b * (y_star ** 2) + c * x_star + d * y_star + e
    
    # Handle negative zero gracefully if needed, or round directly
    x_star = 0.0 if x_star == 0 else round(float(x_star), 6)
    y_star = 0.0 if y_star == 0 else round(float(y_star), 6)
    f_min = 0.0 if f_min == 0 else round(float(f_min), 6)
    
    return {
        "x_star": x_star,
        "y_star": y_star,
        "f_min": f_min
    }
