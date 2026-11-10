import numpy as np

def mahalanobis_distance(x: list, mean: list, cov: list) -> float:
    """
    Returns the Mahalanobis distance as a Python float.
    """
    # Convert inputs to float64 numpy arrays
    x_arr = np.array(x, dtype=np.float64)
    mean_arr = np.array(mean, dtype=np.float64)
    cov_arr = np.array(cov, dtype=np.float64)
    
    # Calculate the difference vector (x - mu)
    diff = x_arr - mean_arr
    
    # Solve the system Cov * y = diff instead of inverting Cov
    y = np.linalg.solve(cov_arr, diff)
    
    # Compute the quadratic form and take the square root
    distance = np.sqrt(np.dot(diff, y))
    
    # Return as a native Python float
    return float(distance)
