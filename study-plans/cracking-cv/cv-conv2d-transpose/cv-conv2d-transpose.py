def conv_transpose_2d(image: list, weight: list, bias: list, stride: int, padding: int) -> list:
    """
    Returns the 2D transposed convolution result as a channel-first nested list.
    
    Shapes:
    - image:  (Cin, H, W)
    - weight: (Cin, Cout, Kh, Kw)
    - bias:   (Cout,)
    - Output: (Cout, Hout, Wout)
    """
    # 1. Extract dimensions
    Cin = len(image)
    H = len(image[0])
    W = len(image[0][0])
    
    Cout = len(weight[0])
    Kh = len(weight[0][0])
    Kw = len(weight[0][0][0])
    
    # 2. Compute intermediate unpadded output dimensions
    # Before cropping via padding, the maximum possible spatial footprint is:
    # H_temp = (H - 1) * stride + Kh
    # W_temp = (W - 1) * stride + Kw
    H_temp = (H - 1) * stride + Kh
    W_temp = (W - 1) * stride + Kw
    
    # 3. Compute final output dimensions (accounting for padding subtraction)
    Hout = H_temp - 2 * padding
    Wout = W_temp - 2 * padding
    
    # 4. Initialize accumulator tensor with zeros
    # Shape: (Cout, H_temp, W_temp)
    accum = [[[0.0 for _ in range(W_temp)] for _ in range(H_temp)] for _ in range(Cout)]
    
    # 5. Scatter-add input contributions
    for c in range(Cin):
        for h in range(H):
            for w in range(W):
                val = image[c][h][w]
                # Calculate the top-left anchor position in the output for this pixel
                out_h_start = h * stride
                out_w_start = w * stride
                
                for o in range(Cout):
                    for kh in range(Kh):
                        for kw in range(Kw):
                            w_val = weight[c][o][kh][kw]
                            accum[o][out_h_start + kh][out_w_start + kw] += val * w_val
                            
    # 6. Apply bias, crop out the padding borders, and round to 4 decimal places
    output = []
    for o in range(Cout):
        channel_out = []
        b = bias[o]
        for i in range(Hout):
            row_out = []
            for j in range(Wout):
                # Map final coordinates (i, j) back to unpadded accumulator coordinates
                accum_h = i + padding
                accum_w = j + padding
                
                # Add bias and round
                final_val = round(accum[o][accum_h][accum_w] + b, 4)
                row_out.append(final_val)
            channel_out.append(channel_out_row := row_out)
        output.append(channel_out)
        
    return output
