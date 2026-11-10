def vector_valued_hessian(x: float, y: float, z: float) -> list:
    """
    Returns the three Hessian matrices as a nested list.
    """
    # Slice H1 for F1 = x^2 * y
    h1 = [
        [round(2.0 * y, 6), round(2.0 * x, 6), 0.0],
        [round(2.0 * x, 6), 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
    
    # Slice H2 for F2 = y^2 * z
    h2 = [
        [0.0, 0.0, 0.0],
        [0.0, round(2.0 * z, 6), round(2.0 * y, 6)],
        [0.0, round(2.0 * y, 6), 0.0]
    ]
    
    # Slice H3 for F3 = x * z^2
    h3 = [
        [0.0, 0.0, round(2.0 * z, 6)],
        [0.0, 0.0, 0.0],
        [round(2.0 * z, 6), 0.0, round(2.0 * x, 6)]
    ]
    
    return [h1, h2, h3]
