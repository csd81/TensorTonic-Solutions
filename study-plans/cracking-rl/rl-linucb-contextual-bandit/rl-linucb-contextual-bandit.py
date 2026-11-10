import numpy as np

def linucb_ucb(A_list: list, b_list: list, context: list, alpha: float) -> list:
    """
    Returns one LinUCB score per arm, rounded to four decimals.
    """
    # Convert shared context to a float64 NumPy array
    x = np.array(context, dtype=np.float64)
    scores = []
    
    # Process each arm individually
    for A_arr, b_arr in zip(A_list, b_list):
        A = np.array(A_arr, dtype=np.float64)
        b = np.array(b_arr, dtype=np.float64)
        
        # Calculate inverse of the regularized Gram matrix
        A_inv = np.linalg.inv(A)
        
        # Estimate coefficients: theta = A_inv * b
        theta = A_inv @ b
        
        # Compute the expected reward (mean) and variance bonus
        mean = np.dot(theta, x)
        var = np.dot(x, A_inv @ x)
        
        # LinUCB score = mean + alpha * sqrt(variance)
        ucb = mean + alpha * np.sqrt(var)
        
        # Append the result rounded to 4 decimal places
        scores.append(round(float(ucb), 4))
        
    return scores
