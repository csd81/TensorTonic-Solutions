def dilated_conv2d(image: list, weight: list, bias: list, dilation: int) -> list:
    """
    Returns the dilated convolution result as a channel-first nested list.
    """
    C_in = len(image)
    H = len(image[0])
    W = len(image[0][0])
    
    C_out = len(weight)
    K_h = len(weight[0][0])
    K_w = len(weight[0][0][0])

    # Calculate output dimensions
    H_out = H - (K_h - 1) * dilation
    W_out = W - (K_w - 1) * dilation

    # Initialize the output tensor with zeros
    output = [[[0.0 for _ in range(W_out)] for _ in range(H_out)] for _ in range(C_out)]

    # Perform dilated cross-correlation
    for o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                # Start with the channel bias
                val = bias[o]
                
                # Accumulate the weighted inputs across channels and kernel dimensions
                for c in range(C_in):
                    for u in range(K_h):
                        for v in range(K_w):
                            # Apply the dilation spacing to the input spatial coordinates
                            in_row = i + u * dilation
                            in_col = j + v * dilation
                            val += weight[o][c][u][v] * image[c][in_row][in_col]
                
                # Round to 4 decimal places and cast to float to prevent integer conversion
                output[o][i][j] = float(round(val, 4))

    return output
