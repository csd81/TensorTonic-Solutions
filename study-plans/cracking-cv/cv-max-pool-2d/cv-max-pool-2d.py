def max_pool_2d(image: list, kernel_size: int, stride: int) -> list:
    """
    Returns the max-pooled 2D list of floats, rounded to four decimals.
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    
    # Calculate output dimensions using the provided formulas
    h_out = ((H - kernel_size) // stride) + 1
    w_out = ((W - kernel_size) // stride) + 1
    
    output = []
    
    for i in range(h_out):
        row_out = []
        # Calculate the starting row index of the window
        r_start = i * stride
        
        for j in range(w_out):
            # Calculate the starting column index of the window
            c_start = j * stride
            
            # Extract elements inside the current k x k window
            max_val = None
            for r in range(r_start, r_start + kernel_size):
                for c in range(c_start, c_start + kernel_size):
                    val = image[r][c]
                    if max_val is None or val > max_val:
                        max_val = val
            
            # Round the maximum value to four decimal places as required
            row_out.append(round(float(max_val), 4))
            
        output.append(row_out)
        
    return output
