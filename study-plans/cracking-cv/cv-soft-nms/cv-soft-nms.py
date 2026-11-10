import math

def soft_nms(boxes: list, scores: list, sigma: float, score_threshold: float) -> list:
    """
    Returns original-index and decayed-score pairs in selection order.
    """
    # Keep track of original indices alongside coordinates and current scores
    # elements: [left, top, right, bottom, score, original_index]
    remaining = []
    for idx, (b, s) in enumerate(zip(boxes, scores)):
        if s > score_threshold:
            remaining.append(b + [float(s), idx])
            
    results = []
    
    while remaining:
        # 1. Select the box with the highest current score.
        # Break ties by lowest original index (remaining[i][5]).
        # To maximize score and minimize index, we use a custom key.
        best_idx = max(range(len(remaining)), key=lambda i: (remaining[i][4], -remaining[i][5]))
        
        b_i = remaining.pop(best_idx)
        bi_left, bi_top, bi_right, bi_bottom, s_i, orig_idx = b_i
        
        # Append to results, rounding only the final score to 4 decimal places
        results.append([orig_idx, round(s_i, 4)])
        
        # 2. Calculate IoU and decay scores for all remaining boxes
        next_remaining = []
        for b_j in remaining:
            bj_left, bj_top, bj_right, bj_bottom, s_j, j_idx = b_j
            
            # Intersection coordinates
            inter_left = max(bi_left, bj_left)
            inter_top = max(bi_top, bj_top)
            inter_right = min(bi_right, bj_right)
            inter_bottom = min(bi_bottom, bj_bottom)
            
            # Continuous coordinates intersection area
            inter_w = max(0.0, inter_right - inter_left)
            inter_h = max(0.0, inter_bottom - inter_top)
            inter_area = inter_w * inter_h
            
            # Box areas
            area_i = max(0.0, bi_right - bi_left) * max(0.0, bi_bottom - bi_top)
            area_j = max(0.0, bj_right - bj_left) * max(0.0, bj_bottom - bj_top)
            
            union_area = area_i + area_j - inter_area
            
            # Calculate IoU (handling zero union constraint)
            iou = inter_area / union_area if union_area > 0 else 0.0
            
            # Apply Gaussian decay
            decay = math.exp(-(iou ** 2) / sigma)
            s_j_new = s_j * decay
            
            # Keep the box only if its updated unrounded score is strictly above the threshold
            if s_j_new > score_threshold:
                b_j[4] = s_j_new
                next_remaining.append(b_j)
                
        remaining = next_remaining
        
    return results
