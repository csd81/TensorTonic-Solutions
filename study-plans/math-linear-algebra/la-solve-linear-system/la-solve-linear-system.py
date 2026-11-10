import numpy as np

def solve_linear_system(A: list, b: list) -> np.ndarray:
    """
    Returns the float64 solution vector.
    """
    # Convert lists to NumPy arrays with float64 data type
    A_arr = np.array(A, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    
    # Solve Ax = b directly and return the solution vector x
    return np.linalg.solve(A_arr, b_arr)
