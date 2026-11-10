import math

def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    """
    Returns the new_h-by-new_w resized list of floats, rounded to four decimals.
    Uses corner-aligned bilinear interpolation mapping.
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    
    output = []
    
    for i in range(new_h):
        row = []
        # Calculate source y coordinate
        if new_h == 1:
            y = 0.0
        else:
            y = i * (H - 1) / (new_h - 1)
            
        # Determine adjacent rows and weights
        y0 = math.floor(y)
        y1 = math.ceil(y)
        
        # Clamp to bounds
        y0 = max(0, min(y0, H - 1))
        y1 = max(0, min(y1, H - 1))
        
        y_weight = y - y0
        
        for j in range(new_w):
            # Calculate source x coordinate
            if new_w == 1:
                x = 0.0
            else:
                x = j * (W - 1) / (new_w - 1)
                
            # Determine adjacent columns and weights
            x0 = math.floor(x)
            x1 = math.ceil(x)
            
            # Clamp to bounds
            x0 = max(0, min(x0, W - 1))
            x1 = max(0, min(x1, W - 1))
            
            x_weight = x - x0
            
            # Fetch pixel values from the 4 corners
            p00 = image[y0][x0]
            p01 = image[y0][x1]
            p10 = image[y1][x0]
            p11 = image[y1][x1]
            
            # Perform bilinear interpolation
            # Interpolate top two pixels along X-axis
            top = p00 + (p01 - p00) * x_weight
            # Interpolate bottom two pixels along X-axis
            bottom = p10 + (p11 - p10) * x_weight
            # Interpolate vertically between top and bottom
            val = top + (bottom - top) * y_weight
            
            row.append(round(val, 4))
        output.append(row)
        
    return output
