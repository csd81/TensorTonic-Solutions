import numpy as np

def morphological_op(image: list, kernel: list, operation: str) -> list:
    """
    Returns the eroded or dilated binary image.
    """
    img = np.array(image, dtype=int)
    ker = np.array(kernel, dtype=int)
    
    kh, kw = ker.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Zero-pad the input image
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Extract sliding windows matching the kernel shape
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    
    # Create a boolean mask of active kernel positions
    mask = (ker == 1)
    
    if operation == "erode":
        # True only if all active kernel positions have a 1 in the window
        result = np.all(windows[:, :, mask] == 1, axis=-1)
    elif operation == "dilate":
        # True if any active kernel position has a 1 in the window
        result = np.any(windows[:, :, mask] == 1, axis=-1)
    else:
        raise ValueError("Operation must be 'erode' or 'dilate'")
        
    return result.astype(int).tolist()
