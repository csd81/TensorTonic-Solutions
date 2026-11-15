import torch

def spread_out_loss(query: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """
    Returns the spread-out regularizer as a 0-dimensional tensor.
    """
    B = query.shape[0]
    
    # Compute Gram matrices (all pairwise inner products) for each tower
    q_gram = query @ query.mT
    p_gram = positive @ positive.mT
    
    # Create a boolean mask to select only off-diagonal elements (i != j)
    off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=query.device)
    
    # Square the off-diagonal inner products and compute their mean for each tower
    q_loss = (q_gram[off_diag_mask] ** 2).mean()
    p_loss = (p_gram[off_diag_mask] ** 2).mean()
    
    return q_loss + p_loss
