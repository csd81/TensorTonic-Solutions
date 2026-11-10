import numpy as np

def vector_projection(u: list, v: list) -> np.ndarray:
    """
    Returns the float64 projection of u onto v.
    """
    # Convert lists to float64 NumPy arrays
    u_arr = np.array(u, dtype=np.float64)
    v_arr = np.array(v, dtype=np.float64)
    
    # Calculate the dot products
    u_dot_v = np.dot(u_arr, v_arr)
    v_dot_v = np.dot(v_arr, v_arr)
    
    # Compute the projection vector
    p = (u_dot_v / v_dot_v) * v_arr
    
    return p
