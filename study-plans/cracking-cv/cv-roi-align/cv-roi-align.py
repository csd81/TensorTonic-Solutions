def roi_align(feature_map: list, boxes: list, output_h: int, output_w: int, sampling_ratio: int) -> list:
    """
    Returns a float list of shape (N, C, output_h, output_w), rounded to 4 decimals.
    """
    C = len(feature_map)
    H = len(feature_map[0])
    W = len(feature_map[0][0])
    N = len(boxes)
    
    # Pre-calculate sampling ratio behavior
    # If sampling_ratio <= 0, typically adaptive sampling is used, but constraints state it is positive.
    # The problem description specifies sy and sx range from 0 to S - 1.
    S = sampling_ratio
    
    output = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Apply the -0.5 coordinate shift specified by "aligned ROI Align"
        x1_prime = x1 - 0.5
        y1_prime = y1 - 0.5
        x2_prime = x2 - 0.5
        y2_prime = y2 - 0.5
        
        box_w = x2_prime - x1_prime
        box_h = y2_prime - y1_prime
        
        box_output = []
        
        for c in range(C):
            channel_map = feature_map[c]
            channel_output = []
            
            for i in range(output_h):
                row_bins = []
                for j in range(output_w):
                    
                    total_val = 0.0
                    total_samples = S * S
                    
                    for sy in range(S):
                        for sx in range(S):
                            # Formula from problem description:
                            # y = y1_prime + (box_h / output_h) * (i + (sy + 0.5) / S)
                            # x = x1_prime + (box_w / output_w) * (j + (sx + 0.5) / S)
                            y = y1_prime + (box_h / output_h) * (i + (sy + 0.5) / S)
                            x = x1_prime + (box_w / output_w) * (j + (sx + 0.5) / S)
                            
                            # Boundary conditions
                            if y < -1.0 or y > H or x < -1.0 or x > W:
                                val = 0.0
                            else:
                                # Clamp to [0, H-1] and [0, W-1] before interpolation
                                y_clamped = max(0.0, min(float(H - 1), y))
                                x_clamped = max(0.0, min(float(W - 1), x))
                                
                                # Bilinear interpolation floors
                                y_low = int(y_clamped)
                                x_low = int(x_clamped)
                                y_high = min(H - 1, y_low + 1)
                                x_high = min(W - 1, x_low + 1)
                                
                                ly = y_clamped - y_low
                                lx = x_clamped - x_low
                                hy = 1.0 - ly
                                hx = 1.0 - lx
                                
                                v1 = channel_map[y_low][x_low]
                                v2 = channel_map[y_low][x_high]
                                v3 = channel_map[y_high][x_low]
                                v4 = channel_map[y_high][x_high]
                                
                                val = (v1 * hy * hx + 
                                       v2 * hy * lx + 
                                       v3 * ly * hx + 
                                       v4 * ly * lx)
                                
                            total_val += val
                    
                    # Round each bin average to 4 decimal places
                    bin_avg = round(total_val / total_samples, 4)
                    row_bins.append(bin_avg)
                    
                channel_output.append(row_bins)
            box_output.append(channel_output)
        output.append(box_output)
        
    return output
