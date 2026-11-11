import numpy as np

def nms(boxes: list, scores: list, iou_threshold: float) -> dict:
    """
    Applies greedy non-maximum suppression for axis-aligned boxes.
    
    Returns a dictionary with kept_indices and kept_scores.
    """
    if not boxes:
        return {"kept_indices": [], "kept_scores": []}
    
    # Cast to float64 for precise coordinate calculations
    boxes = np.array(boxes, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    
    # Extract coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate vectorised areas (no +1 added to width/height for continuous coordinates)
    areas = (x2 - x1) * (y2 - y1)
    
    # Track original indices to return them properly
    indices = np.arange(len(boxes))
    
    # Sort: Descending by score, break ties by ascending original index.
    # np.lexsort sorts by the last key first. Providing (indices, -scores) 
    # sorts by -scores ascending (meaning scores descending) primary, 
    # and indices ascending secondary.
    order = np.lexsort((indices, -scores))
    
    kept_indices = []
    kept_scores = []
    
    while len(order) > 0:
        # Pick the box with the highest priority score/index tier
        current_idx = order[0]
        kept_indices.append(int(current_idx))
        kept_scores.append(round(float(scores[current_idx]), 4))
        
        if len(order) == 1:
            break
            
        remaining_indices = order[1:]
        
        # Calculate intersection boundaries against all remaining boxes
        ix1 = np.maximum(x1[current_idx], x1[remaining_indices])
        iy1 = np.maximum(y1[current_idx], y1[remaining_indices])
        ix2 = np.minimum(x2[current_idx], x2[remaining_indices])
        iy2 = np.minimum(y2[current_idx], y2[remaining_indices])
        
        # Intersection width and height
        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h
        
        # Union Area calculation
        union_area = areas[current_idx] + areas[remaining_indices] - inter_area
        
        # Safely compute IoU, avoiding division by zero if union area is zero
        iou = np.zeros_like(union_area)
        non_zero_union = union_area > 0
        iou[non_zero_union] = inter_area[non_zero_union] / union_area[non_zero_union]
        
        # Strictly suppress elements where IoU > threshold
        # Retain elements where IoU <= threshold
        mask = iou <= iou_threshold
        order = remaining_indices[mask]
        
    return {"kept_indices": kept_indices, "kept_scores": kept_scores}
