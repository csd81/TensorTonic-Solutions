import numpy as np

def matrix_factorization_sgd_step(U: list, V: list, r: float, lr: float, reg: float) -> list:
    """
    Returns the updated user and item vectors in a two-item list.
    """
    u = np.array(U, dtype=float)
    v = np.array(V, dtype=float)
    
    # 1. Compute the prediction error
    error = r - np.dot(u, v)
    
    # 2. Update both vectors simultaneously using original values
    u_new = u + lr * (error * v - reg * u)
    v_new = v + lr * (error * u - reg * v)
    
    # 3. Round to 4 decimal places and convert back to standard Python lists
    return [
        np.round(u_new, 4).tolist(),
        np.round(v_new, 4).tolist()
    ]
