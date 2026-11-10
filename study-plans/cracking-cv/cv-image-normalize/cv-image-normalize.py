def normalize_image(image: list, mean: list, std: list) -> list:
    """
    Returns the normalized H-by-W-by-C list of floats, rounded to four decimals.
    """
    H = len(image)
    W = len(image[0])
    C = len(image[0][0])
    
    # Initialize a clean nested structure for the output
    normalized = [[[0.0] * C for _ in range(W)] for _ in range(H)]
    
    for h in range(H):
        for w in range(W):
            for c in range(C):
                # Apply the normalization formula: (X - mu) / sigma
                val = (image[h][w][c] - mean[c]) / std[c]
                # Round to 4 decimal places as required
                normalized[h][w][c] = round(val, 4)
                
    return normalized
