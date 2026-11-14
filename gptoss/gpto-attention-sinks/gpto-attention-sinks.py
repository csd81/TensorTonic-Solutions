import torch

def softmax_with_sinks(scores: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """
    Returns attention weights with the same shape as scores.
    """
    H, n_q, _ = scores.shape
    
    # Expand sinks from (H,) to (H, n_q, 1) to align with every query row
    sink_col = sinks.view(H, 1, 1).expand(H, n_q, 1)
    
    # Concatenate along the key dimension -> shape (H, n_q, n_k + 1)
    logits = torch.cat([scores, sink_col], dim=-1)
    
    # PyTorch's softmax handles large negative masks (-1e9) stably via log-sum-exp shifting
    weights = torch.softmax(logits, dim=-1)
    
    # Drop the sink column to return weights of shape (H, n_q, n_k)
    return weights[..., :-1]
