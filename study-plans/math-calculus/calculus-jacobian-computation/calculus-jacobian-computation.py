def jacobian_3d(x: float, y: float, z: float) -> list:
    """
    Returns the 3 by 3 Jacobian as a nested list.
    """
    row0 = [float(2 * x * y), float(x**2), 0.0]
    row1 = [0.0, float(2 * y), 1.0]
    row2 = [float(z**2), 0.0, float(2 * x * z)]
    
    return [row0, row1, row2]
