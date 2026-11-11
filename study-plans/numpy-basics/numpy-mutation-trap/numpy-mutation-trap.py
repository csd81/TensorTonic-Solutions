import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    """
    Returns a (2, n) float64 array: original row, then clipped row.
    """
    # Extract the target row and convert it to a float64 NumPy array
    orig_row = np.array(data[row_idx], dtype=np.float64)
    
    # Create the clipped row without altering the original row array
    clipped_row = np.clip(orig_row, lo, hi)
    
    # Stack them vertically into a (2, n) shape array
    return np.vstack((orig_row, clipped_row))
