import numpy as np

def lbfgs_direction(grad: list, s_list: list, y_list: list) -> list:
    """
    Returns the L-BFGS descent direction (-r) from the stored history.
    """
    # Convert inputs to numpy arrays for efficient vector operations
    q = np.array(grad, dtype=float)
    s = [np.array(v, dtype=float) for v in s_list]
    y = [np.array(v, dtype=float) for v in y_list]
    
    m = len(s)
    alpha = [0.0] * m
    rho = [0.0] * m
    
    # 1. Backward Pass (Newest to Oldest)
    for i in reversed(range(m)):
        rho[i] = 1.0 / np.dot(y[i], s[i])
        alpha[i] = rho[i] * np.dot(s[i], q)
        q -= alpha[i] * y[i]
        
    # 2. Base Hessian Scaling (using the newest history pair, index m-1)
    gamma = np.dot(s[-1], y[-1]) / np.dot(y[-1], y[-1])
    r = gamma * q
    
    # 3. Forward Pass (Oldest to Newest)
    for i in range(m):
        beta = rho[i] * np.dot(y[i], r)
        r += s[i] * (alpha[i] - beta)
        
    # The search direction is the negated final vector (-r)
    descent_direction = -r
    
    return descent_direction.tolist()
