import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    # Convert input to a 64-bit float NumPy array
    A_arr = np.array(A, dtype=np.float64)
    n = A_arr.shape[0]
    
    # Form the augmented matrix [A | I]
    aug = np.hstack((A_arr, np.eye(n, dtype=np.float64)))
    
    for col in range(n):
        # 1. Partial pivoting: find row with largest absolute value in current column
        pivot_row = col + np.argmax(np.abs(aug[col:, col]))
        
        # 2. Check for singularity (pivot below tolerance)
        if np.abs(aug[pivot_row, col]) < 1e-12:
            return None
            
        # 3. Swap current row with pivot row
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]
            
        # 4. Normalize pivot row so the pivot becomes 1
        aug[col] = aug[col] / aug[col, col]
        
        # 5. Eliminate current column from all other rows (both above and below)
        for row in range(n):
            if row != col:
                factor = aug[row, col]
                aug[row] -= factor * aug[col]
                
    # Return the right half of the augmented matrix
    return aug[:, n:]
