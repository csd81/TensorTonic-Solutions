import numpy as np

def sobel_edges(image: list) -> dict:
    """
    Returns gx, gy, and magnitude as H-by-W float lists, rounded to four decimals.
    """
    # Convert input to numpy float64 array
    img = np.array(image, dtype=np.float64)
    H, W = img.shape
    
    # Define the kernels as specified
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)
    
    # Apply zero-padding of 1 pixel on all sides
    padded_img = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    
    # Initialize output arrays
    gx_arr = np.zeros_like(img)
    gy_arr = np.zeros_like(img)
    
    # Perform cross-correlation
    for i in range(H):
        for j in range(W):
            # Extract the 3x3 local neighborhood centered at (i, j) in the original image
            # In the padded image, this window shifts by +1 in both dimensions
            window = padded_img[i:i+3, j:j+3]
            
            # Cross-correlation is element-wise multiplication followed by summation
            gx_arr[i, j] = np.sum(window * Kx)
            gy_arr[i, j] = np.sum(window * Ky)
            
    # Compute the unrounded magnitude
    magnitude_arr = np.sqrt(gx_arr**2 + gy_arr**2)
    
    # Round values to 4 decimal places and convert back to standard Python nested lists
    res_gx = np.round(gx_arr, 4).tolist()
    res_gy = np.round(gy_arr, 4).tolist()
    res_magnitude = np.round(magnitude_arr, 4).tolist()
    
    return {
        "gx": res_gx,
        "gy": res_gy,
        "magnitude": res_magnitude
    }
