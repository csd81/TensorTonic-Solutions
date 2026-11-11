def histogram_equalize(image: list) -> list:
    """
    Returns the histogram-equalized grayscale image.
    """
    # 1. Compute the histogram for intensities 0 through 255
    hist = [0] * 256
    total_pixels = 0
    for row in image:
        for val in row:
            hist[val] += 1
            total_pixels += 1
            
    # 2. Compute the Cumulative Distribution Function (CDF)
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
        
    # 3. Find the smallest non-zero value in the CDF
    cdf_min = next(val for val in cdf if val > 0)
    
    # 4. Handle the edge case where all pixels have the same intensity
    denominator = total_pixels - cdf_min
    if denominator == 0:
        return [[0 for _ in row] for row in image]
        
    # 5. Precompute a lookup table (LUT) for all 256 intensity values
    lut = [0] * 256
    for i in range(256):
        if hist[i] > 0:
            lut[i] = round(((cdf[i] - cdf_min) / denominator) * 255)
            
    # 6. Map each pixel in the image to its new intensity value
    return [[lut[val] for val in row] for row in image]
