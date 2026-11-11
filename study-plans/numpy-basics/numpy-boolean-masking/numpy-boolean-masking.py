import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    # Convert the input data to a NumPy array of type float64
    arr = np.array(data, dtype=np.float64)
    
    # 1. Create a boolean mask where elements are strictly greater than the threshold
    gt_mask = arr > threshold
    
    # Layer 1: Element mask (1.0 if True, 0.0 if False)
    layer1 = gt_mask.astype(np.float64)
    
    # 2. Check conditions along each row (axis=1)
    any_row_condition = np.any(gt_mask, axis=1, keepdims=True)  # Shape (m, 1)
    all_row_condition = np.all(gt_mask, axis=1, keepdims=True)  # Shape (m, 1)
    
    # Layer 2 & 3: Use np.where to broadcast the row-wise conditions and zero-out rows
    layer2 = np.where(any_row_condition, arr, 0.0)
    layer3 = np.where(all_row_condition, arr, 0.0)
    
    # Stack the three layers along a new first axis to get a shape of (3, m, n)
    return np.stack([layer1, layer2, layer3], axis=0)
