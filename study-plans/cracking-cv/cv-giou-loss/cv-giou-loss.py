def giou_loss(pred_boxes: list, target_boxes: list) -> float:
    """
    Returns the mean GIoU loss as a float, rounded to 4 decimals.
    """
    total_loss = 0.0
    n = len(pred_boxes)
    eps = 1e-7

    for i in range(n):
        # Unpack prediction and target boxes [x1, y1, x2, y2]
        px1, py1, px2, py2 = pred_boxes[i]
        tx1, ty1, tx2, ty2 = target_boxes[i]
        
        # 1. Calculate Areas of individual boxes
        pred_area = (px2 - px1) * (py2 - py1)
        target_area = (tx2 - tx1) * (ty2 - ty1)
        
        # 2. Calculate Intersection coordinates and area
        ix1 = max(px1, tx1)
        iy1 = max(py1, ty1)
        ix2 = min(px2, tx2)
        iy2 = min(py2, ty2)
        
        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        intersection = inter_w * inter_h
        
        # 3. Calculate Union area
        union = pred_area + target_area - intersection
        
        # 4. Calculate IoU
        iou = intersection / (union + eps)
        
        # 5. Calculate Enclosing Box (C / A_i) coordinates and area
        cx1 = min(px1, tx1)
        cy1 = min(py1, ty1)
        cx2 = max(px2, tx2)
        cy2 = max(py2, ty2)
        
        enclosing_area = (cx2 - cx1) * (cy2 - cy1)
        
        # 6. Calculate GIoU and the Loss for this pair
        giou = iou - (enclosing_area - union) / (enclosing_area + eps)
        loss = 1.0 - giou
        
        total_loss += loss

    # Return the mean loss rounded to 4 decimals
    return round(total_loss / n, 4)
