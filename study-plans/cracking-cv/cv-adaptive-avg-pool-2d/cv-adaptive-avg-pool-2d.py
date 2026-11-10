import math

def adaptive_avg_pool_2d(image: list, output_h: int, output_w: int) -> list:
    """
    Returns the output_h-by-output_w average list of floats, rounded to four decimals.
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    
    output = []
    
    for q_h in range(output_h):
        # Row intervals using the specified formulas
        r_start = math.floor((q_h * H) / output_h)
        r_end = math.ceil(((q_h + 1) * H) / output_h)
        
        row_out = []
        for q_w in range(output_w):
            # Column intervals using the specified formulas
            c_start = math.floor((q_w * W) / output_w)
            c_end = math.ceil(((q_w + 1) * W) / output_w)
            
            # Extract values from the specified bounding box and calculate average
            total_sum = 0.0
            count = 0
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    total_sum += image[r][c]
                    count += 1
            
            # Round final values to four decimal places
            row_out.append(round(total_sum / count, 4))
            
        output.append(row_out)
        
    return output
