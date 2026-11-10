def avg_pool_2d(image: list, kernel_size: int, stride: int) -> list:
    """
    Returns the average-pooled 2D list of floats, rounded to four decimals.
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    
    # Calculate output dimensions using the floor formula
    h = ((H - kernel_size) // stride) + 1
    w = ((W - kernel_size) // stride) + 1
    
    # Total number of elements inside a square window
    window_elements = kernel_size * kernel_size
    
    output = []
    
    # Slide across the height rows
    for i in range(h):
        row_out = []
        r_start = i * stride
        
        # Slide across the width columns
        for j in range(w):
            c_start = j * stride
            
            # Extract and sum window elements
            window_sum = 0.0
            for r in range(r_start, r_start + kernel_size):
                for c in range(c_start, c_start + kernel_size):
                    window_sum += image[r][c]
            
            # Compute average and round to 4 decimal places
            avg_val = round(window_sum / window_elements, 4)
            row_out.append(avg_val)
            
        output.append(row_out)
        
    return output
