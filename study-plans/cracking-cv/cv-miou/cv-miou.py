import torch

def mean_iou(pred: list, target: list, num_classes: int) -> float:
    """
    Returns the mean IoU over present classes as a float, rounded to 4 decimals.
    """
    # Convert inputs to PyTorch tensors of type int64
    pred_tensor = torch.tensor(pred, dtype=torch.int64)
    target_tensor = torch.tensor(target, dtype=torch.int64)
    
    ious = []
    
    # Evaluate each class independently
    for c in range(num_classes):
        # Create boolean masks for current class
        pred_c = (pred_tensor == c)
        target_c = (target_tensor == c)
        
        # Check if the class is present in either the prediction or the target
        if not pred_c.any() and not target_c.any():
            continue
            
        # Calculate intersection and union
        intersection = torch.logical_and(pred_c, target_c).sum().item()
        union = torch.logical_or(pred_c, target_c).sum().item()
        
        # Calculate IoU for the current class
        iou_c = intersection / union if union > 0 else 0.0
        ious.append(iou_c)
        
    # If no classes were present in either mask, return 0.0
    if not ious:
        return 0.0
        
    # Calculate the mean over all present classes and round to 4 decimals
    return round(sum(ious) / len(ious), 4)
