import numpy as np

def sigmoid_squeeze_analysis(x: list) -> dict:
    """
    Returns sigmoid bounds and saturation flags in a dictionary.
    """
    # Convert input list to a NumPy float64 array as required
    x_arr = np.array(x, dtype=np.float64)
    
    # Compute the core sigmoid function components
    sigmoid = 1.0 / (1.0 + np.exp(-x_arr))
    
    # Calculate the theoretical upper and lower squeeze bounds
    lower_bound = np.maximum(0.0, 1.0 - np.exp(-x_arr))
    upper_bound = np.minimum(1.0, np.exp(x_arr))
    
    # Combine lower, sigmoid, and upper values into an N x 3 matrix
    # Rounding is not strictly enforced in the output format but formatting to 6 decimal places match standard test cases.
    bounds_matrix = np.column_stack((lower_bound, sigmoid, upper_bound))
    
    # Determine gradient saturation: smaller of σ(x) and 1 - σ(x) < 0.0001
    smaller_val = np.minimum(sigmoid, 1.0 - sigmoid)
    is_saturated = smaller_val < 0.0001
    
    # Convert the structures back to standard Python lists matching requirements
    return {
        "bounds": bounds_matrix.tolist(),
        "is_saturated": is_saturated.tolist()
    }
