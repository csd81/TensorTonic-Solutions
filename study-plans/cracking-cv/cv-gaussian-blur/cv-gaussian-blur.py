import numpy as np

def gaussian_blur_2d(image: list, kernel_size: int, sigma: float) -> list:
    """
    Returns the H-by-W blurred list of floats, rounded to four decimals.
    Handles extreme sigma values (both near-zero and near-infinity) gracefully.
    """
    image_np = np.array(image, dtype=np.float64)
    H, W = image_np.shape
    
    # 1. Generate the 2D Gaussian kernel coordinates
    c = (kernel_size - 1) / 2
    
    # 2. Compute unnormalized weights safely
    if sigma < 1e-150:
        # Extreme Underflow: Delta function (center pixel gets 1.0)
        K = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        K[int(c), int(c)] = 1.0
    elif sigma > 1e150:
        # Extreme Overflow: Uniform distribution (all pixels share equal weight)
        K = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size * kernel_size)
    else:
        # Normal range computation
        a, b = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size), indexing='ij')
        denom = 2 * (sigma ** 2)
        g = np.exp(-((a - c)**2 + (b - c)**2) / denom)
        K = g / np.sum(g)
        
    # 3. Apply zero padding to each side of the image
    pad_width = int(c)
    padded_image = np.pad(image_np, pad_width, mode='constant', constant_values=0)
    
    # 4. Perform cross-correlation
    output = np.zeros_like(image_np)
    for i in range(H):
        for j in range(W):
            region = padded_image[i : i + kernel_size, j : j + kernel_size]
            output[i, j] = np.sum(region * K)
            
    # 5. Round final values to 4 decimal places and convert back to standard Python list
    return np.round(output, 4).tolist()
