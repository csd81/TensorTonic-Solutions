import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    # Calculate mean along the batch dimension (dim=0)
    mean = X.mean(dim=0)
    
    # Calculate population variance along the batch dimension (dim=0)
    # setting correction=0 specifies population variance (N) instead of sample variance (N-1)
    var = X.var(dim=0, correction=0)
    
    # Standardize the features
    X_normalized = (X - mean) / torch.sqrt(var + eps)
    
    # Apply scale (gamma) and shift (beta)
    Y = gamma * X_normalized + beta
    
    return Y
