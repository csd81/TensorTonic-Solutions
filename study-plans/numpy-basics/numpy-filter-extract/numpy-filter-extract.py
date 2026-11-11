import numpy as np

def filter_and_extract(data: list, row_start: int, row_stop: int, threshold: float) -> np.ndarray:
    """
    Returns matching values in row-major order as a 1D float64 array.
    """
    # Convert input to a NumPy array with the requested float64 type
    arr = np.array(data, dtype=np.float64)
    
    # Slice the requested rows using standard Python slicing
    sliced_rows = arr[row_start:row_stop]
    
    # Create a boolean mask for elements strictly greater than the threshold
    mask = sliced_rows > threshold
    
    # Applying the mask automatically flattens the array in row-major (C-style) order
    return sliced_rows[mask]
