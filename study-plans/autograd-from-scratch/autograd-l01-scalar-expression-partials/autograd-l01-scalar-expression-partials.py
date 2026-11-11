import numpy as np

def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    a_val = np.float64(a)
    b_val = np.float64(b)
    c_val = np.float64(c)
    h_val = np.float64(h)

    def expr(x: np.float64, y: np.float64, z: np.float64) -> np.float64:
        return x * y + z

    d = expr(a_val, b_val, c_val)
    
    partial_a = (expr(a_val + h_val, b_val, c_val) - d) / h_val
    partial_b = (expr(a_val, b_val + h_val, c_val) - d) / h_val
    partial_c = (expr(a_val, b_val, c_val + h_val) - d) / h_val

    return (float(d), float(partial_a), float(partial_b), float(partial_c))
