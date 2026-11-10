def nms(boxes: list, scores: list, iou_threshold: float) -> list:
    """
    Returns the retained original box indices in selection order.
    """
    if not boxes:
        return []

    # Pair each box index with its score and precalculate area
    # Coordinates format: [left, top, right, bottom] -> [x1, y1, x2, y2]
    box_data = []
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        box_data.append({
            'idx': idx,
            'score': score,
            'box': box,
            'area': area
        })
    
    # Sort boxes:
    # 1. Primary key: highest score first (-x['score'])
    # 2. Secondary key: lowest original index first (x['idx'])
    box_data.sort(key=lambda x: (-x['score'], x['idx']))
    
    keep = []
    
    while box_data:
        # Select the highest-scoring remaining box
        current = box_data.pop(0)
        keep.append(current['idx'])
        
        remaining = []
        for item in box_data:
            # Unpack coordinates
            ax1, ay1, ax2, ay2 = current['box']
            bx1, by1, bx2, by2 = item['box']
            
            # Find the intersection bounds
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            
            # Continuous coordinate intersection area
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = float(inter_w * inter_h)
            
            union_area = current['area'] + item['area'] - inter_area
            
            # Compute IoU safely
            if union_area == 0.0:
                iou = 0.0
            else:
                iou = inter_area / union_area
            
            # A selected box suppresses another box only when IoU is STRICTLY greater
            if iou <= iou_threshold:
                remaining.append(item)
                
        box_data = remaining
        
    return keep
