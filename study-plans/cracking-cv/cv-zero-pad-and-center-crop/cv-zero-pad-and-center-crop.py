def pad_and_center_crop(image: list, pad: int, crop_h: int, crop_w: int) -> list:
    # Get original image dimensions
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    
    # Calculate padded dimensions
    padded_H = H + 2 * pad
    padded_W = W + 2 * pad
    
    # Create the padded image filled with 0.0
    padded_image = [[0.0] * padded_W for _ in range(padded_H)]
    
    # Copy the original image into the center of the padded image
    for r in range(H):
        for c in range(W):
            padded_image[r + pad][c + pad] = float(image[r][c])
            
    # Calculate the starting top-left coordinates for the center crop
    r_start = (H + 2 * pad - crop_h) // 2
    c_start = (W + 2 * pad - crop_w) // 2
    
    # Extract the cropped region and round values to 4 decimal places
    cropped_image = []
    for r in range(r_start, r_start + crop_h):
        row = []
        for c in range(c_start, c_start + crop_w):
            row.append(round(padded_image[r][c], 4))
        cropped_image.append(row)
        
    return cropped_image
