import torch

def topk_accuracy(logits: list, targets: list, k: int) -> float:
    """
    Returns the top-k accuracy as a float, rounded to 4 decimals with stable tie-breaking.
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    targets_tensor = torch.tensor(targets, dtype=torch.int64)
    
    # Use torch.argsort with stable=True to preserve index order during ties
    sorted_indices = torch.argsort(logits_tensor, dim=1, descending=True, stable=True)
    topk_indices = sorted_indices[:, :k]
    
    # Broadcast the comparison across the top-k indices
    targets_reshaped = targets_tensor.view(-1, 1)
    matches = (topk_indices == targets_reshaped)
    
    # Calculate accuracy
    correct_count = matches.any(dim=1).sum().item()
    total_samples = len(targets)
    
    return round(correct_count / total_samples, 4)
