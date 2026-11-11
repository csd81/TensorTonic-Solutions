import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    # First convert the data to a NumPy array with float64 data type
    arr = np.array(data, dtype=np.float64)
    
    if operation == "flatten":
        return arr.flatten()
    elif operation == "transpose":
        return arr.T
    elif operation == "add_batch":
        return arr.reshape(1, *arr.shape)
    else:
        raise ValueError(f"Unknown operation: {operation}")
