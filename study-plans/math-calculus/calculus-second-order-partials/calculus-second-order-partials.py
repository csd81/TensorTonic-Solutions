import numpy as np

def second_order_partials(x: float, y: float) -> dict:
    """
    Returns the two pure second-order partial derivatives.
    """
    sin_xy = np.sin(x * y)
    
    # Calculate the pure second-order partial derivatives
    d2f_dx2 = -float(y**2 * sin_xy)
    d2f_dy2 = -float(x**2 * sin_xy)
    
    return {
        "d2f_dx2": d2f_dx2,
        "d2f_dy2": d2f_dy2
    }
