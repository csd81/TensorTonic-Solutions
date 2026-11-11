import numpy as np

def quantize_and_frame(data: list, decimals: int, pad_width: int) -> np.ndarray:
    """
    Returns float64 slices of rounded, floored, and ceiling-rounded values with zero borders.
    """
    # 1. Convert the input data to a base float64 NumPy array
    arr = np.array(data, dtype=np.float64)
    
    # 2. Compute the three independent rounding modes
    rounded = np.round(arr, decimals)
    floored = np.floor(arr)
    ceiled = np.ceil(arr)
    
    # 3. Stack the three layers along a new first axis (axis=0)
    stacked = np.stack([rounded, floored, ceiled], axis=0)
    
    # 4. Pad only the spatial dimensions (axis 1 and axis 2) with zeros
    # Axis 0 (the 3 layers) receives no padding: (0, 0)
    padded = np.pad(
        stacked, 
        pad_width=((0, 0), (pad_width, pad_width), (pad_width, pad_width)), 
        mode='constant', 
        constant_values=0
    )
    
    return padded
