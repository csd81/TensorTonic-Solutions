import math

def ciou_loss(pred_boxes: list, target_boxes: list) -> float:
    """
    Returns the mean CIoU loss as a float, rounded to 4 decimals.
    """
    eps = 1e-7
    total_loss = 0.0
    N = len(pred_boxes)
    
    for p, t in zip(pred_boxes, target_boxes):
        px1, py1, px2, py2 = p
        tx1, ty1, tx2, ty2 = t
        
        # Dimensions
        pw, ph = px2 - px1, py2 - py1
        tw, th = tx2 - tx1, ty2 - ty1
        
        p_area = pw * ph
        t_area = tw * th
        
        # Intersection Area
        ix1 = max(px1, tx1)
        iy1 = max(py1, ty1)
        ix2 = min(px2, tx2)
        iy2 = min(py2, ty2)
        
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        
        # IoU
        union = p_area + t_area - inter
        iou = inter / (union + eps)
        
        # Center Points & Squared Distance (rho^2)
        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        
        # Smallest Enclosing Box Diagonal (d^2)
        cx1 = min(px1, tx1)
        cy1 = min(py1, ty1)
        cx2 = max(px2, tx2)
        cy2 = max(py2, ty2)
        cw, ch = cx2 - cx1, cy2 - cy1
        d2 = cw ** 2 + ch ** 2
        
        # Aspect Ratio Penalty (v) and Weight (alpha)
        v = (4.0 / (math.pi ** 2)) * (math.atan(tw / th) - math.atan(pw / ph)) ** 2
        alpha = v / (1.0 - iou + v + eps)
        
        # CIoU Loss component
        loss = (1.0 - iou) + (rho2 / (d2 + eps)) + alpha * v
        total_loss += loss
        
    return round(total_loss / N, 4)
