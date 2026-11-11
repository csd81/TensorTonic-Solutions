import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    # Find the maximum value along each row (dim=1)
    # keepdim=True ensures the output shape is (N, 1) for proper broadcasting
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    
    # Subtract the maximum for numerical stability
    stable_logits = logits - max_logits
    
    # Exponentiate the stabilized logits
    exp_logits = torch.exp(stable_logits)
    
    # Sum the exponents along each row (dim=1)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    
    # Divide to get the final probability distribution
    return exp_logits / sum_exp
