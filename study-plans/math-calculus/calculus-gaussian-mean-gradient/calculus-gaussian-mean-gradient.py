import numpy as np

def gaussian_log_mean_gradient(x: list, mu: list, Sigma: list) -> list:
    """
    Returns the log-density gradient with respect to the mean.
    """
    # 1. Convert input structures to float64 NumPy arrays
    x_arr = np.asarray(x, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    Sigma_arr = np.asarray(Sigma, dtype=np.float64)
    
    # 2. Form the residual vector (x - mu)
    residual = x_arr - mu_arr
    
    # 3. Solve the linear system Sigma * g = residual instead of computing Sigma^-1
    grad = np.linalg.solve(Sigma_arr, residual)
    
    # 4. Round values to 6 decimal places and return as a Python list
    return [round(float(val), 6) for val in grad]
