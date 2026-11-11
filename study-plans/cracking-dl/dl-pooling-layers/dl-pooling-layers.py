import numpy as np

def pooling(input: list, pool_size: int, stride: int, pool_type: str) -> list:
    """
    Returns the pooled channel-first tensor.
    """
    # Convert input to a NumPy array for easier slicing
    arr = np.array(input, dtype=np.float64)
    C, H, W = arr.shape
    
    # Calculate spatial output dimensions
    H_out = int(np.floor((H - pool_size) / stride)) + 1
    W_out = int(np.floor((W - pool_size) / stride)) + 1
    
    # Initialize the output array
    out = np.zeros((C, H_out, W_out), dtype=np.float64)
    
    # Perform independent pooling per channel and window
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                # Calculate window boundaries
                r_start = i * stride
                r_end = r_start + pool_size
                c_start = j * stride
                c_end = c_start + pool_size
                
                window = arr[c, r_start:r_end, c_start:c_end]
                
                # Apply the specified pooling type
                if pool_type == "max":
                    out[c, i, j] = np.max(window)
                elif pool_type == "average":
                    out[c, i, j] = np.mean(window)
                    
    # Round to 4 decimal places and convert back to a standard Python nested list
    return np.round(out, 4).tolist()
