def pairwise_iou(boxes1: list, boxes2: list) -> list:
    """
    Returns the N-by-M pairwise IoU matrix as a nested list.
    """
    matrix = []
    
    for box1 in boxes1:
        row = []
        l1, t1, r1, b1 = box1
        # Precompute area for box1
        a1 = (r1 - l1) * (b1 - t1)
        
        for box2 in boxes2:
            l2, t2, r2, b2 = box2
            # Precompute area for box2
            a2 = (r2 - l2) * (b2 - t2)
            
            # Find coordinates of the intersection rectangle
            inter_l = max(l1, l2)
            inter_t = max(t1, t2)
            inter_r = min(r1, r2)
            inter_b = min(b1, b2)
            
            # Compute width and height of intersection, clamp at 0
            inter_w = max(0.0, inter_r - inter_l)
            inter_h = max(0.0, inter_b - inter_t)
            
            inter_area = inter_w * inter_h
            union_area = a1 + a2 - inter_area
            
            # If union area is zero (degenerate/empty boxes), IoU is 0.0
            if union_area <= 0:
                iou = 0.0
            else:
                iou = inter_area / union_area
                
            row.append(round(iou, 4))
            
        matrix.append(row)
        
    return matrix
