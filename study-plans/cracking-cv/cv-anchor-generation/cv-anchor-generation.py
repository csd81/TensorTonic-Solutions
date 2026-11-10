import math

def generate_anchors(feature_h: int, feature_w: int, stride: int, base_size: int, scales: list, aspect_ratios: list) -> list:
    """
    Returns the ordered anchor boxes as a list of four-float lists.
    """
    anchors = []
    
    for fy in range(feature_h):
        for fx in range(feature_w):
            # Compute anchor centers
            cx = (fx + 0.5) * stride
            cy = (fy + 0.5) * stride
            
            for scale in scales:
                for ratio in aspect_ratios:
                    # Calculate dimensions based on base size, scale, and width-to-height aspect ratio
                    w = base_size * scale * math.sqrt(ratio)
                    h = base_size * scale / math.sqrt(ratio)
                    
                    # Convert center and dimensions to [left, top, right, bottom]
                    left = round(cx - w / 2, 4)
                    top = round(cy - h / 2, 4)
                    right = round(cx + w / 2, 4)
                    bottom = round(cy + h / 2, 4)
                    
                    anchors.append([left, top, right, bottom])
                    
    return anchors
